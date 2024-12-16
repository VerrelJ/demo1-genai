from google.cloud import discoveryengine
from google.cloud import storage
from google.cloud import firestore
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair, GroundingSource, ChatMessage, ChatSession
import json

from google.cloud import documentai
from google.cloud import discoveryengine

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate

from langchain_google_community._utils import get_client_info

import logging
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain_community.document_loaders.gcs_file import GCSFileLoader
from langchain_community.utilities.vertexai import get_client_info
import re


from google.cloud import storage
from google.cloud import aiplatform
from google.cloud import documentai
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint

from IPython.display import display, HTML
import markdown as md


from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores.vectorstores import VectorSearchVectorStore

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community import VertexAIRank

from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain.docstore.document import Document
from langchain_core.runnables import chain

from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate

from langchain_google_community import VertexAICheckGroundingWrapper

from rich import print

PROJECT_ID = "capable-conduit-442309-v8"
REGION = "asia-southeast2"
LOCATION_SEARCH = 'global'
DATABASE_NAME = 'demo1-genai'
DATA_STORE_ID = 'document-ingestion_1709564596074'
BUCKET_NAME = "demo1-genai"
GCS_BUCKET_URI = "gs://demo1-genai"  # @param {type:"string"}
GCS_OUTPUT_PATH = f"{GCS_BUCKET_URI}"  # DocAI Layout Parser Output Path
GCS_BUCKET_NAME = GCS_BUCKET_URI.replace("gs://", "")
VS_INDEX_NAME = "demo1-index"  # @param {type:"string"}
VS_INDEX_ID = "5670911540161675264"
VS_INDEX_ENDPOINT_NAME = "demo1-index-endpoint"  # @param {type:"string"}
VS_INDEX_ENDPOINT_ID = "4395865080034492416"
VS_CONTENTS_DELTA_URI = f"{GCS_BUCKET_URI}/index/embeddings"
VS_DIMENSIONS = 768
VS_APPROX_NEIGHBORS = 150
VS_INDEX_UPDATE_METHOD = "STREAM_UPDATE"
VS_INDEX_SHARD_SIZE = "SHARD_SIZE_SMALL"
VS_LEAF_NODE_EMB_COUNT = 500
VS_LEAF_SEARCH_PERCENT = 80
VS_DISTANCE_MEASURE_TYPE = "DOT_PRODUCT_DISTANCE"
VS_MACHINE_TYPE = "e2-standard-16"
VS_MIN_REPLICAS = 1
VS_MAX_REPLICAS = 1
VS_DESCRIPTION = "Index for DIY RAG with Vertex AI APIs"  # @param {type:"string"}
EMBEDDINGS_MODEL_NAME = "textembedding-gecko-multilingual"
LLM_MODEL_NAME = "gemini-1.5-flash-002"
DOCAI_LOCATION = "us"  # @param ["us", "eu"]
DOCAI_PROCESSOR_NAME = "projects/688888656210/locations/us/processors/19a4d4f04e1be9fa"
CREATE_RESOURCES = False  # @param {type:"boolean"}
# flag to run data ingestion
RUN_INGESTION = True  # @param {type:"boolean"}
vertexai.init(project=PROJECT_ID, location=REGION)
if TYPE_CHECKING:
    from google.api_core.operation import Operation  # type: ignore[import]
    from google.cloud.documentai import (  # type: ignore[import]
        DocumentProcessorServiceClient,
    )


logger = logging.getLogger(__name__)


@dataclass
class DocAIParsingResults:
    """A dataclass to store Document AI parsing results."""

    source_path: str
    parsed_path: str


class DocAIParser(BaseBlobParser):
    """`Google Cloud Document AI` parser.

    For a detailed explanation of Document AI, refer to the product documentation.
    https://cloud.google.com/document-ai/docs/overview
    """

    def __init__(
        self,
        *,
        client: Optional["DocumentProcessorServiceClient"] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        gcs_output_path: Optional[str] = None,
        processor_name: Optional[str] = None,
    ) -> None:
        """Initializes the parser.

        Args:
            client: a DocumentProcessorServiceClient to use
            location: a Google Cloud location where a Document AI processor is located
            gcs_output_path: a path on Google Cloud Storage to store parsing results
            processor_name: full resource name of a Document AI processor or processor
                version

        You should provide either a client or location (and then a client
            would be instantiated).
        """

        if bool(client) == bool(location):
            raise ValueError(
                "You must specify either a client or a location to instantiate "
                "a client."
            )

        pattern = r"projects\/[0-9]+\/locations\/[a-z\-0-9]+\/processors\/[a-z0-9]+"
        if processor_name and not re.fullmatch(pattern, processor_name):
            raise ValueError(
                f"Processor name {processor_name} has the wrong format. If your "
                "prediction endpoint looks like https://us-documentai.googleapis.com"
                "/v1/projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID:process,"
                " use only projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID "
                "part."
            )

        self._gcs_output_path = gcs_output_path
        self._processor_name = processor_name
        if client:
            self._client = client
        else:
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud.documentai import DocumentProcessorServiceClient
            except ImportError as exc:
                raise ImportError(
                    "Could not import google-cloud-documentai python package. "
                    "Please, install docai dependency group: "
                    "`pip install langchain-google-community[docai]`"
                ) from exc
            options = ClientOptions(
                quota_project_id=project_id,
                api_endpoint=f"{location}-documentai.googleapis.com",
            )
            self._client = DocumentProcessorServiceClient(
                client_options=options,
                client_info=get_client_info(module="document-ai"),
            )
            # get processor type
            self._processor_type = self._client.get_processor(name=processor_name).type
            if self._processor_type == "LAYOUT_PARSER_PROCESSOR":
                self._use_layout_parser = True
            else:
                self._use_layout_parser = False

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parses a blob lazily.

        Args:
            blobs: a Blob to parse

        This is a long-running operation. A recommended way is to batch
            documents together and use the `batch_parse()` method.
        """
        yield from self.batch_parse([blob], gcs_output_path=self._gcs_output_path)

    def online_process(
        self,
        blob: Blob,
        enable_native_pdf_parsing: bool = True,
        field_mask: Optional[str] = None,
        page_range: Optional[List[int]] = None,
        chunk_size: int = 500,
        include_ancestor_headings: bool = True,
    ) -> Iterator[Document]:
        """Parses a blob lazily using online processing.

        Args:
            blob: a blob to parse.
            enable_native_pdf_parsing: enable pdf embedded text extraction
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"
            page_range: list of page numbers to parse. If `None`,
                entire document will be parsed.
            chunk_size: the maximum number of characters per chunk
            include_ancestor_headings: whether to include ancestor headings in the chunks
                https://cloud.google.com/document-ai/docs/reference/rpc/google.cloud.documentai.v1beta3#chunkingconfig
        """
        try:
            from google.cloud import documentai
            from google.cloud.documentai_v1.types import (  # type: ignore[import, attr-defined]
                OcrConfig,
                ProcessOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-documentai python package. "
                "Please, install docai dependency group: "
                "`pip install langchain-google-community[docai]`"
            ) from exc
        try:
            from google.cloud.documentai_toolbox.wrappers.page import (  # type: ignore[import]
                _text_from_layout,
            )
        except ImportError as exc:
            raise ImportError(
                "documentai_toolbox package not found, please install it with "
                "`pip install langchain-google-community[docai]`"
            ) from exc

        if self._use_layout_parser:
            layout_config = ProcessOptions.LayoutConfig(
                chunking_config=ProcessOptions.LayoutConfig.ChunkingConfig(
                    chunk_size=chunk_size,
                    include_ancestor_headings=include_ancestor_headings,
                )
            )
            individual_page_selector = (
                ProcessOptions.IndividualPageSelector(pages=page_range)
                if page_range
                else None
            )
            process_options = ProcessOptions(
                layout_config=layout_config,
                individual_page_selector=individual_page_selector,
            )
        else:
            ocr_config = (
                OcrConfig(enable_native_pdf_parsing=enable_native_pdf_parsing)
                if enable_native_pdf_parsing
                else None
            )
            individual_page_selector = (
                ProcessOptions.IndividualPageSelector(pages=page_range)
                if page_range
                else None
            )
            process_options = ProcessOptions(
                ocr_config=ocr_config, individual_page_selector=individual_page_selector
            )

        response = self._client.process_document(
            documentai.ProcessRequest(
                name=self._processor_name,
                gcs_document=documentai.GcsDocument(
                    gcs_uri=blob.path,
                    mime_type=blob.mimetype or "application/pdf",
                ),
                process_options=process_options,
                skip_human_review=True,
                field_mask=field_mask,
            )
        )

        if self._use_layout_parser:
            yield from (
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "source": blob.path,
                    },
                )
                for chunk in response.document.chunked_document.chunks
            )
        else:
            yield from (
                Document(
                    page_content=_text_from_layout(page.layout, response.document.text),
                    metadata={
                        "page": page.page_number,
                        "source": blob.path,
                    },
                )
                for page in response.document.pages
            )

    def batch_parse(
        self,
        blobs: Sequence[Blob],
        gcs_output_path: Optional[str] = None,
        timeout_sec: int = 3600,
        check_in_interval_sec: int = 60,
        chunk_size: int = 500,
        include_ancestor_headings: bool = True,
    ) -> Iterator[Document]:
        """Parses a list of blobs lazily.

        Args:
            blobs: a list of blobs to parse.
            gcs_output_path: a path on Google Cloud Storage to store parsing results.
            timeout_sec: a timeout to wait for Document AI to complete, in seconds.
            check_in_interval_sec: an interval to wait until next check
                whether parsing operations have been completed, in seconds
        This is a long-running operation. A recommended way is to decouple
            parsing from creating LangChain Documents:
            >>> operations = parser.docai_parse(blobs, gcs_path)
            >>> parser.is_running(operations)
            You can get operations names and save them:
            >>> names = [op.operation.name for op in operations]
            And when all operations are finished, you can use their results:
            >>> operations = parser.operations_from_names(operation_names)
            >>> results = parser.get_results(operations)
            >>> docs = parser.parse_from_results(results)
        """
        output_path = gcs_output_path or self._gcs_output_path
        if not output_path:
            raise ValueError(
                "An output path on Google Cloud Storage should be provided."
            )
        operations = self.docai_parse(
            blobs,
            gcs_output_path=output_path,
            chunk_size=chunk_size,
            include_ancestor_headings=include_ancestor_headings,
        )
        operation_names = [op.operation.name for op in operations]
        logger.debug(
            "Started parsing with Document AI, submitted operations %s", operation_names
        )
        time_elapsed = 0
        while self.is_running(operations):
            time.sleep(check_in_interval_sec)
            time_elapsed += check_in_interval_sec
            if time_elapsed > timeout_sec:
                raise TimeoutError(
                    "Timeout exceeded! Check operations " f"{operation_names} later!"
                )
            logger.debug(".")

        results = self.get_results(operations=operations)
        yield from self.parse_from_results(results)

    def parse_from_results(
        self, results: List[DocAIParsingResults]
    ) -> Iterator[Document]:
        try:
            from google.cloud.documentai_toolbox.utilities.gcs_utilities import (  # type: ignore[import]
                split_gcs_uri,
            )
            from google.cloud.documentai_toolbox.wrappers.document import (  # type: ignore[import]
                _get_shards,
            )
            from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "documentai_toolbox package not found, please install it with "
                "`pip install langchain-google-community[docai]`"
            ) from exc
        #tuning biar cepet
        for result in results:
            print(f"processing: {result.parsed_path}")
            gcs_bucket_name, gcs_prefix = split_gcs_uri(result.parsed_path)
            shards = _get_shards(gcs_bucket_name, gcs_prefix + "/")
            if self._use_layout_parser:
                yield from (
                    Document(
                        page_content=chunk.content,
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "source": result.source_path,
                        },
                    )
                    for shard in shards
                    for chunk in shard.chunked_document.chunks
                )
            else:
                yield from (
                    Document(
                        page_content=_text_from_layout(page.layout, shard.text),
                        metadata={
                            "page": page.page_number,
                            "source": result.source_path,
                        },
                    )
                    for shard in shards
                    for page in shard.pages
                )

    def operations_from_names(self, operation_names: List[str]) -> List["Operation"]:
        """Initializes Long-Running Operations from their names."""
        try:
            from google.longrunning.operations_pb2 import (  # type: ignore[import]
                GetOperationRequest,
            )
        except ImportError as exc:
            raise ImportError(
                "long running operations package not found, please install it with"
                "`pip install langchain-google-community[docai]`"
            ) from exc

        return [
            self._client.get_operation(request=GetOperationRequest(name=name))
            for name in operation_names
        ]

    def is_running(self, operations: List["Operation"]) -> bool:
        return any(not op.done() for op in operations)

    def docai_parse(
        self,
        blobs: Sequence[Blob],
        *,
        gcs_output_path: Optional[str] = None,
        processor_name: Optional[str] = None,
        batch_size: int = 1000,
        enable_native_pdf_parsing: bool = True,
        field_mask: Optional[str] = None,
        chunk_size: Optional[int] = 500,
        include_ancestor_headings: Optional[bool] = True,
    ) -> List["Operation"]:
        """Runs Google Document AI PDF Batch Processing on a list of blobs.

        Args:
            blobs: a list of blobs to be parsed
            gcs_output_path: a path (folder) on GCS to store results
            processor_name: name of a Document AI processor.
            batch_size: amount of documents per batch
            enable_native_pdf_parsing: a config option for the parser
            field_mask: a comma-separated list of which fields to include in the
                Document AI response.
                suggested: "text,pages.pageNumber,pages.layout"
            chunking_config: Serving config for chunking when using layout
                parser processor. Specify config parameters as dictionary elements.
                https://cloud.google.com/document-ai/docs/reference/rpc/google.cloud.documentai.v1beta3#chunkingconfig

        Document AI has a 1000 file limit per batch, so batches larger than that need
        to be split into multiple requests.
        Batch processing is an async long-running operation
        and results are stored in a output GCS bucket.
        """
        try:
            from google.cloud import documentai
            from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions
        except ImportError as exc:
            raise ImportError(
                "documentai package not found, please install it with "
                "`pip install langchain-google-community[docai]`"
            ) from exc

        output_path = gcs_output_path or self._gcs_output_path
        if output_path is None:
            raise ValueError(
                "An output path on Google Cloud Storage should be provided."
            )
        processor_name = processor_name or self._processor_name
        if processor_name is None:
            raise ValueError("A Document AI processor name should be provided.")

        operations = []
        for batch in batch_iterate(size=batch_size, iterable=blobs):
            input_config = documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(
                    documents=[
                        documentai.GcsDocument(
                            gcs_uri=blob.path,
                            mime_type=blob.mimetype or "application/pdf",
                        )
                        for blob in batch
                    ]
                )
            )

            output_config = documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=output_path, field_mask=field_mask
                )
            )

            if self._use_layout_parser:
                layout_config = ProcessOptions.LayoutConfig(
                    chunking_config=ProcessOptions.LayoutConfig.ChunkingConfig(
                        chunk_size=chunk_size,
                        include_ancestor_headings=include_ancestor_headings,
                    )
                )
                process_options = ProcessOptions(layout_config=layout_config)
            else:
                process_options = (
                    ProcessOptions(
                        ocr_config=OcrConfig(
                            enable_native_pdf_parsing=enable_native_pdf_parsing
                        )
                    )
                    if enable_native_pdf_parsing
                    else None
                )
            operations.append(
                self._client.batch_process_documents(
                    documentai.BatchProcessRequest(
                        name=processor_name,
                        input_documents=input_config,
                        document_output_config=output_config,
                        process_options=process_options,
                        skip_human_review=True,
                    )
                )
            )
        return operations

    def get_results(self, operations: List["Operation"]) -> List[DocAIParsingResults]:
        try:
            from google.cloud.documentai_v1 import (  # type: ignore[import]
                BatchProcessMetadata,
            )
        except ImportError as exc:
            raise ImportError(
                "documentai package not found, please install it with "
                "`pip install langchain-google-community[docai]`"
            ) from exc

        return [
            DocAIParsingResults(
                source_path=status.input_gcs_source,
                parsed_path=status.output_gcs_destination,
            )
            for op in operations
            for status in (
                op.metadata.individual_process_statuses
                if isinstance(op.metadata, BatchProcessMetadata)
                else BatchProcessMetadata.deserialize(
                    op.metadata.value
                ).individual_process_statuses
            )
        ]
    
class CustomGCSDirectoryLoader(GCSDirectoryLoader, BaseLoader):
    def load(self, file_pattern=None) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-storage python package. "
                "Please install it with `pip install google-cloud-storage`."
            )
        client = storage.Client(
            project=self.project_name,
            client_info=get_client_info(module="google-cloud-storage"),
        )

        regex = None
        if file_pattern:
            regex = re.compile(r'{}'.format(file_pattern))

        docs = []
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            # we shall just skip directories since GCSFileLoader creates
            # intermediate directories on the fly
            if blob.name.endswith("/"):
                continue
            if regex and not regex.match(blob.name):
                continue
            # Use the try-except block here
            try:
                logger.info(f"Processing {blob.name}")
                temp_blob = Blob(path=f"gs://{blob.bucket.name}/{blob.name}")
                docs.append(temp_blob)
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(f"Problem processing blob {blob.name}, message: {e}")
                    continue
                else:
                    raise e
        return docs
    

def get_batches(items: List, n: int = 1000) -> List[List]:
    n = max(1, n)
    return [items[i : i + n] for i in range(0, len(items), n)]


def add_data(vector_store, chunks) -> None:
    if RUN_INGESTION:
        batch_size = 1000
        texts = get_batches([chunk.page_content for chunk in chunks], n=batch_size)
        metadatas = get_batches([chunk.metadata for chunk in chunks], n=batch_size)

        for i, (b_texts, b_metadatas) in enumerate(zip(texts, metadatas)):
            print(f"Adding {len(b_texts)} data points to index")
            is_complete_overwrite = bool(i == 0)
            vector_store.add_texts(
                texts=b_texts,
                metadatas=b_metadatas,
                is_complete_overwrite=is_complete_overwrite,
            )
    else:
        print("Skipping ingestion. Enable `RUN_INGESTION` flag")


def display_grounded_generation(response) -> None:
    # Extract the answer with citations and cited chunks
    answer_with_citations = response.answer_with_citations
    cited_chunks = response.cited_chunks

    # Build HTML for the chunks
    chunks_html = "".join(
        [
            f"<div id='chunk-{index}' class='chunk'>"
            + f"<div class='source'>Source {index}: <a href='{chunk['source'].metadata['source']}' target='_blank'>{chunk['source'].metadata['source']}</a></div>"
            + f"<p>{chunk['chunk_text']}</p>"
            + "</div>"
            for index, chunk in enumerate(cited_chunks)
        ]
    )

    # Replace citation indices with hoverable spans
    for index in range(len(cited_chunks)):
        answer_with_citations = answer_with_citations.replace(
            f"[{index}]",
            f"<span class='citation' onmouseover='highlight({index})' onmouseout='unhighlight({index})'>[{index}]</span>",
        )

    # The complete HTML
    html_content = f"""
    <style>
    .answer-box {{
        background-color: #f8f9fa;
        border-left: 4px solid #0056b3;
        padding: 20px;
        margin-bottom: 20px;
        color: #000;
    }}
    .citation {{
        background-color: transparent;
        cursor: pointer;
    }}
    .chunk {{
        background-color: #ffffff;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin-bottom: 10px;
        transition: background-color 0.3s;
        color: #000;
    }}
    .source {{
        font-weight: bold;
        margin-bottom: 5px;
    }}
    a {{
        text-decoration: none;
        color: #0056b3;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class='answer-box'>{answer_with_citations}</div>
    <div class='chunks-box'>{chunks_html}</div>
    <script>
    function highlight(index) {{
        // Highlight the citation in the answer
        document.querySelectorAll('.citation').forEach(function(citation) {{
            if (citation.textContent === '[' + index + ']') {{
                citation.style.backgroundColor = '#ffff99';
            }}
        }});
        // Highlight the corresponding chunk
        document.getElementById('chunk-' + index).style.backgroundColor = '#ffff99';
    }}
    function unhighlight(index) {{
        // Unhighlight the citation in the answer
        document.querySelectorAll('.citation').forEach(function(citation) {{
            if (citation.textContent === '[' + index + ']') {{
                citation.style.backgroundColor = 'transparent';
            }}
        }});
        // Unhighlight the corresponding chunk
        document.getElementById('chunk-' + index).style.backgroundColor = '#ffffff';
    }}
    </script>
    """
    display(HTML(html_content))


def clear_docai_output(username):
    bucket = client_gcs.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=f"{username}/docai_output/")
    for blob in blobs:
        blob.delete()

def get_docs(username):
    loader = CustomGCSDirectoryLoader(
        project_name=PROJECT_ID,
        bucket="demo1-genai",
        prefix=f"{username}/docs",
    )
    doc_blobs = loader.load(file_pattern=".")

    parser = DocAIParser(
        project_id=PROJECT_ID,
        location=DOCAI_LOCATION,
        processor_name=DOCAI_PROCESSOR_NAME,
        # gcs_output_path=f"{GCS_BUCKET_URI}/{username}/docai_output",
    )

    docs = list(
        parser.batch_parse(
            doc_blobs,  # filter only last 40 for docs after 2020
            chunk_size=500,
            include_ancestor_headings=True,
            gcs_output_path=f"{GCS_BUCKET_URI}/{username}/docai_output"
        )
    )
    return docs

def get_docs_from_bucket(username):
    loader = CustomGCSDirectoryLoader(
        project_name=PROJECT_ID,
        bucket="demo1-genai",
        prefix=f"{username}/docai_output",
    )
    doc_blobs = loader.load(file_pattern=".")

    return doc_blobs


def update_vector_store():
    vector_store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=GCS_BUCKET_NAME,
        index_id=VS_INDEX_ID,
        endpoint_id=VS_INDEX_ENDPOINT_ID,
        embedding=VertexAIEmbeddings(model_name=EMBEDDINGS_MODEL_NAME),
        stream_update=True,
    )
    return vector_store

# add_data(vector_store, docs)

def retriever(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever


def create_answer():
    llm = VertexAI(model_name="gemini-1.5-pro-001", max_output_tokens=1024)
    template = """
    Anda adalah Asisten Virtual AI yang didedikasikan untuk memberikan informasi berdasarkan context document yang diberikan.
 
    Misi Anda:
    - Menjawab pertanyaan dari pengguna dengan memberikan respons sesuai dengan konteks dan basis pengetahuan yang telah diberikan dan ditentukan.
    - Basis pengetahuan Anda adalah satu-satunya sumber informasi. Jangan menjawab pertanyaan di luar konteks.
    - JANGAN PERNAH MEMBERIKAN LINK DOKUMEN. LANGSUNG JAWAB BERDASARKAN ISI DOKUMEN SAJA!!!
    - Jika diminta untuk berikan analisa, berikan jawaban dalam bentuk terstruktur sehingga dapat mempermudah user dalam membaca hasil analisa
    - Pada proses analisa, pastikan analisa dilakukan dengan context dan grounding yang diberikan serta penalaranmu dalam menyelesaikan masalah tersebut
    
    Bahasa:
    - Bahasa yang digunakan adalah bahasa indonesia dan bahasa inggris, jika context dan grounding yang diberikan dalam bahasa inggris, maka jawab dengan bahasa inggris dan jika context dan grounding yang diberikan dalam bahasa indonesia, maka jawab dengan bahasa indonesia
    - Jawaban menggunakan bahasa yang formal dan profesional.
    - Jangan menggunakan bahasa gaul dari bahasa apa pun dan kata slang dari bahasa apa pun.
    - Jika terdapat atau menggunakan istilah selain dari bahasa indonesia, berikan arti dan penjelasan singkat
    
    Basis Pengetahuan:
    - Jawablah pertanyaan hanya dengan basis pengetahuan yang kamu diberikan pada context dan grounding, serta informasi pada percakapan chat sebelumnya
    - Jika terdapat informasi atau pengetahuan yang kurang, mintakan kepada user untuk memberikan informasi atau pengetahuan tersebut.
    
    Dokumen yang Dilampirkan:{context}

    Question:
    {query}
    """
    prompt = PromptTemplate.from_template(template)

    create_answer = prompt | llm

    return create_answer


def output_parser():
    output_parser = VertexAICheckGroundingWrapper(
        project_id=PROJECT_ID,
        location_id="global",
        grounding_config="default_grounding_config",
        top_n=3,
    )
    return output_parser

# @chain
# def check_grounding_output_parser(answer_candidate: str, documents: List[Document], output_parser):
#     docs_dict = {"documents": documents}
#     return output_parser.with_config(configurable=docs_dict).invoke(
#         answer_candidate
#     )

@chain
def qa_with_check_grounding(input_dict):
    query = input_dict["query"]
    create_answer = input_dict["create_answer"]
    retriever = input_dict["retriever"]
    output_parser = input_dict["output_parser"]
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "query": RunnablePassthrough()}
    )
    docs = setup_and_retrieval.invoke(query)
    answer_candidate = create_answer.invoke(docs)
    docs_dict = {"documents": docs["context"]}
    check_grounding_output = output_parser.with_config(configurable=docs_dict).invoke(
        answer_candidate
    )
    return check_grounding_output


### I use this when developing in local, it faster than developing in cloud run. So, comment these lines when deploying ###

# from google.oauth2 import service_account

# with open('mii-telkomsel-genai-poc-da5fa004e23f.json') as source:
#     info = json.load(source)

# credentials = service_account.Credentials.from_service_account_info(info)
# client_search = discoveryengine.DocumentServiceClient(credentials=credentials)
# client_gcs = storage.Client(project=PROJECT_ID, credentials=credentials)
# client_db = firestore.Client(project=PROJECT_ID, database=DATABASE_NAME, credentials=credentials)
# vertexai.init(project=PROJECT_ID, location=LOCATION_VERTEX_AI, credentials=credentials)

### Instead use these lines when deploying in Cloud Run ###

client_search = discoveryengine.DocumentServiceClient()
client_gcs = storage.Client(project=PROJECT_ID)
client_db = firestore.Client(project=PROJECT_ID, database=DATABASE_NAME)


# parent = client_search.branch_path(
#         project=PROJECT_ID,
#         location=LOCATION_SEARCH,
#         data_store=DATA_STORE_ID,
#         branch="default_branch",
#     )

# def get_data(username):
#     temp_array = []
    
#     response_search = client_search.list_documents(parent=parent)
#     for result in response_search:
#         split_result = result.content.uri.split('/')
#         print(split_result[-2], username)
#         if split_result[-2] == username:
#             temp_dict = {}
#             temp_dict['Document Name'] = split_result[-1]
#             temp_dict['Document Location'] = result.content.uri
#             temp_dict['Submited?'] = '✅ Yes'
#             temp_array.append(temp_dict)
    
#     print(temp_array)

#     response_gcs = client_gcs.list_blobs(BUCKET_NAME, prefix=f"{username}/")
#     for result in response_gcs:
#         if result.name.split('/')[-1] == '':
#             continue
#         print(f'gs://{BUCKET_NAME}/{result.name}')
#         if any(x['Document Location'] == f'gs://{BUCKET_NAME}/{result.name}' for x in temp_array):
#             continue
#         temp_dict = {}
#         temp_dict['Document Name'] = result.name.split('/')[-1]
#         temp_dict['Document Location'] = f"gs://{BUCKET_NAME}/{username}/{result.name}"
#         temp_dict['Submited?'] = '❌ No'
#         temp_array.append(temp_dict)

#     return temp_array


def upload_files(files, username):
    for name in files:
        bucket = client_gcs.get_bucket(BUCKET_NAME)
        blob = bucket.blob(f"{username}/docs/" + name.name)
        blob.upload_from_string(name.getvalue())

# def refresh_vertex_search(username):
#     request = discoveryengine.ImportDocumentsRequest(
#         parent=parent,
#         gcs_source=discoveryengine.GcsSource(
#             input_uris=[f"gs://{BUCKET_NAME}/{username}/*"], data_schema="content"
#         ),
#         reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
#         # reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
#     )

#     operation = client_search.import_documents(request=request)

#     print(f"Waiting for operation to complete: {operation.operation.name}")
#     response = operation.result()
#     metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)
#     print(response)
#     print(metadata)

def get_prompt_for_context():
    prompt = """
            You are an intelligent assistant helping the users with their questions on {{company | research papers | …}}. Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.
            
            Do not try to make up an answer:
             - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
             - If the context is empty, just say "I do not know the answer to that."
            
            CONTEXT: 
            {{retrieved_information}}
            
            QUESTION:
            {{question}
            
            Helpful Answer:
            """
    return prompt

# def initialize_parameter_bot():
#     grounding_value = GroundingSource.VertexAISearch(data_store_id=DATA_STORE_ID, location=LOCATION_SEARCH, project=PROJECT_ID)

#     parameters = {
#       "candidate_count": 1,
#       "grounding_source": grounding_value,
#       "max_output_tokens": 1024,
#       "temperature": 0.1,
#       "top_k": 40,
#       "top_p": 1
#       }

#     return parameters

# def initialize_chat_bot():
#   chat_model = ChatModel.from_pretrained("chat-bison")
#   chat = chat_model.start_chat(
#       context=get_prompt_for_context(),
#       max_output_tokens = 1024,
#       temperature = 0.1,
#       top_p = 1
#       )
#   return chat

def authentication(username, password):
    query_res = client_db.collection("demo1-genai")\
                .where(filter=firestore.FieldFilter("username", "==", username))\
                .where(filter=firestore.FieldFilter("password", "==", password))\
                .get()
    return query_res


def list_files_in_bucket(username):
    """Lists all the files in the specified GCS bucket."""
    from google.cloud import storage
    
    bucket = client_gcs.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=f"{username}/docs/")
    
    file_names = []
    for blob in blobs:
        file_name = blob.name.split('/')[-1]
        file_names.append(file_name)
        
    return file_names
