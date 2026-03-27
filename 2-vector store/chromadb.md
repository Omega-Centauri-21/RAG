```python
## langchain imports

# from langchain.text_splitter import RecursiveCharacterTextSplitter # old
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

## vectorstores
from langchain_community.vectorstores import Chroma

## utility imports
import numpy as np
from typing import List
```

    d:\Learning\RAG\.venv\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

## 1. Sample data


```python
## create sample documents
sample_docs = [
    """Machine Learning Basics

    Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programming. It revolves around three key components — Task (T), Experience (E), and Performance (P) — where models improve performance on a task through experience over time.

    Core Types of ML include:
    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).
    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.
    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.
    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.
    """,

    """
    Deep Learning Essentials

    Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to automatically learn complex patterns from large datasets. Inspired by the human brain, these networks consist of input layers, hidden layers, and output layers, where each neuron applies nonlinear transformations to extract features and make predictions.
    Key differences from traditional machine learning include automated feature extraction, end-to-end learning, and the ability to handle unstructured data like images, audio, and text. However, deep learning requires large datasets, high computational power (GPUs/TPUs), and often suffers from interpretability challenges.
    """,

    """
    Large Language Models

    Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text .
    Core Functionality: LLMs work by predicting the next token (word or subword) in a sequence based on context. This enables them to perform tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs such as images or audio when extended into Large Multimodal Models (LMMs)
    """
]

sample_docs
```




    ['Machine Learning Basics\n\n    Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programming. It revolves around three key components — Task (T), Experience (E), and Performance (P) — where models improve performance on a task through experience over time.\n\n    Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.\n    ',
     '\n    Deep Learning Essentials\n\n    Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to automatically learn complex patterns from large datasets. Inspired by the human brain, these networks consist of input layers, hidden layers, and output layers, where each neuron applies nonlinear transformations to extract features and make predictions.\n    Key differences from traditional machine learning include automated feature extraction, end-to-end learning, and the ability to handle unstructured data like images, audio, and text. However, deep learning requires large datasets, high computational power (GPUs/TPUs), and often suffers from interpretability challenges.\n    ',
     '\n    Large Language Models\n\n    Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text .\n    Core Functionality: LLMs work by predicting the next token (word or subword) in a sequence based on context. This enables them to perform tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs such as images or audio when extended into Large Multimodal Models (LMMs)\n    ']




```python
### save sample docs
import tempfile
temp_dir = tempfile.mkdtemp()

for i, doc in enumerate(sample_docs):
    print(f"path: {temp_dir}/doc_{i}.txt")
    with open(f"{temp_dir}/doc_{i}.txt", "w", encoding="utf-8") as f:
        f.write(doc)

print(f"Sample document create in : {temp_dir}")
```

    path: C:\Users\NILANJ~1.PAU\AppData\Local\Temp\tmpgi6jnjye/doc_0.txt
    path: C:\Users\NILANJ~1.PAU\AppData\Local\Temp\tmpgi6jnjye/doc_1.txt
    path: C:\Users\NILANJ~1.PAU\AppData\Local\Temp\tmpgi6jnjye/doc_2.txt
    Sample document create in : C:\Users\NILANJ~1.PAU\AppData\Local\Temp\tmpgi6jnjye
    


```python
temp_dir
```




    'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye'



### 2. Document loader


```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    temp_dir,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf'},
    show_progress=True
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(f"\nFirst document preview:")
print(documents[0].page_content[:200] + "...")
```

    100%|██████████| 3/3 [00:00<00:00, 583.33it/s]

    Loaded 3 documents
    
    First document preview:
    Machine Learning Basics
    
        Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programmin...
    

    
    

### Document Splitting


```python
## Initialize the text Splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Max for each chunk
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""] # Reirarchy of Separators
)

chunks = text_splitter.split_documents(documents)
```


```python
chunks
```




    [Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_0.txt'}, page_content='Machine Learning Basics\n\n    Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programming. It revolves around three key components — Task (T), Experience (E), and Performance (P) — where models improve performance on a task through experience over time.'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_0.txt'}, page_content='Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_1.txt'}, page_content='Deep Learning Essentials'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_1.txt'}, page_content='Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to automatically learn complex patterns from large datasets. Inspired by the human brain, these networks consist of input layers, hidden layers, and output layers, where each neuron applies nonlinear transformations to extract features and make predictions.'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_1.txt'}, page_content='Key differences from traditional machine learning include automated feature extraction, end-to-end learning, and the ability to handle unstructured data like images, audio, and text. However, deep learning requires large datasets, high computational power (GPUs/TPUs), and often suffers from interpretability challenges.'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Large Language Models'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text .'),
     Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Core Functionality: LLMs work by predicting the next token (word or subword) in a sequence based on context. This enables them to perform tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs such as images or audio when extended into Large Multimodal Models (LMMs)')]



## Embedding Models


```python
from langchain_huggingface import HuggingFaceEmbeddings

## Initialize a simple Embedding model (no API Key needed!)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # model_name="sentence-transformers/paraphrase-multilingual-MinLM-L12-v2",
    multi_process=True,
    show_progress=True,
    cache_folder="./../model_cache/",
    # model_kwargs = {"device": "cpu"}
    # model_kwargs = {"device": "gpu"}
    # model_kwargs = {"device": "cuda"}
)
embedding_model
```




    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_folder='./../model_cache/', model_kwargs={}, encode_kwargs={}, query_encode_kwargs={}, multi_process=True, show_progress=True)




```python
sample_text = "Machine Learning is facinating"
vector = embedding_model.embed_query(sample_text)
vector
```




    [0.037038762122392654,
     -0.089223712682724,
     0.04945294186472893,
     -0.004823900293558836,
     -0.060524024069309235,
     0.0006581124616786838,
     0.04959085211157799,
     -0.01670856401324272,
     -0.07168750464916229,
     0.06141171604394913,
     -0.02796000987291336,
     0.04393816739320755,
     0.01350246649235487,
     0.015536176040768623,
     -0.10310302674770355,
     -0.023365292698144913,
     -0.0045269508846104145,
     -0.044909995049238205,
     0.012581870891153812,
     -0.03896067291498184,
     -0.04681098461151123,
     -0.005554227624088526,
     -0.007222628220915794,
     0.006900689098984003,
     0.05475478991866112,
     0.043456193059682846,
     0.012755880132317543,
     -0.010721025057137012,
     0.06398898363113403,
     -0.024143526330590248,
     -0.03364863991737366,
     0.03451896831393242,
     -0.010241327807307243,
     0.007147525437176228,
     -0.05561354383826256,
     -0.0030984908808022738,
     0.07359659671783447,
     0.06452351063489914,
     -0.0047429585829377174,
     0.007473898120224476,
     -0.019505254924297333,
     0.003111850470304489,
     -0.00032042182283475995,
     0.003723275614902377,
     0.06317473948001862,
     0.026456521824002266,
     -0.013684891164302826,
     -0.12782657146453857,
     0.024365805089473724,
     0.04805993288755417,
     -0.12011224031448364,
     -0.10923632234334946,
     -0.04206587001681328,
     -0.03917970135807991,
     -0.035883888602256775,
     -0.020674332976341248,
     0.02473599649965763,
     0.05556667968630791,
     -0.019547317177057266,
     -0.016565430909395218,
     -0.03540952503681183,
     -0.051106009632349014,
     -0.0689857229590416,
     -0.000979939941316843,
     0.13350264728069305,
     -0.017404932528734207,
     -0.07877328246831894,
     0.050595223903656006,
     0.024854909628629684,
     0.016644693911075592,
     0.027125896885991096,
     -0.03422122448682785,
     0.03540781885385513,
     0.1068406030535698,
     0.0782187357544899,
     0.047810330986976624,
     0.05801050737500191,
     -0.0071020969189703465,
     0.10184263437986374,
     0.02152133919298649,
     0.0442153736948967,
     0.06379378587007523,
     0.013992932625114918,
     -0.005784083157777786,
     0.03374111279845238,
     -0.10452646017074585,
     -0.010800851508975029,
     -0.004791391082108021,
     -0.0018142462940886617,
     -0.021405085921287537,
     0.06460662931203842,
     -0.025862988084554672,
     -0.12527363002300262,
     0.02387753687798977,
     0.022784782573580742,
     -0.024389365687966347,
     -0.05725078657269478,
     0.01716461591422558,
     -0.0023624487221240997,
     0.0642896443605423,
     -0.06192799285054207,
     0.029044071212410927,
     -0.030370669439435005,
     -0.01219435315579176,
     -0.06333083659410477,
     -0.05986644700169563,
     0.07885122299194336,
     0.011306819505989552,
     0.10862836241722107,
     -0.08881956338882446,
     -0.0241928081959486,
     0.04932316765189171,
     -0.002328674541786313,
     -0.01370923686772585,
     -0.016305873170495033,
     0.05400697514414787,
     0.04227369651198387,
     0.019771981984376907,
     0.01717870868742466,
     0.10562069714069366,
     -0.022122327238321304,
     0.0442042350769043,
     -0.005602257326245308,
     0.015284758992493153,
     0.04750944674015045,
     -0.1010713055729866,
     -0.07164985686540604,
     -1.9489944292473153e-34,
     -0.05931485816836357,
     0.03542868793010712,
     -0.0054261148907244205,
     0.10475761443376541,
     0.009309565648436546,
     -0.06617569923400879,
     0.010040473192930222,
     0.009304424747824669,
     0.04639606550335884,
     -0.012680836953222752,
     0.048193491995334625,
     0.10647968202829361,
     -0.006678509991616011,
     0.00682454090565443,
     0.04914311692118645,
     -0.1107226237654686,
     -0.035506248474121094,
     0.026498591527342796,
     0.014652238227427006,
     -0.0841861292719841,
     0.004935984034091234,
     -0.054180506616830826,
     0.053303491324186325,
     -0.12313349545001984,
     -0.079398974776268,
     0.025091255083680153,
     0.07222998142242432,
     0.06333041936159134,
     0.09141616523265839,
     0.06535357236862183,
     0.04094180837273598,
     0.015015220269560814,
     -0.1327478289604187,
     -0.07621867209672928,
     0.015548245050013065,
     -0.015063881874084473,
     0.07893572002649307,
     -0.06268474459648132,
     0.07476688921451569,
     -0.036116212606430054,
     -0.05266990140080452,
     -0.014002561569213867,
     0.0756683424115181,
     -0.03666774928569794,
     0.023690903559327126,
     0.04735013097524643,
     -0.0013825965579599142,
     -0.1366100013256073,
     -0.011487297713756561,
     -0.0267441738396883,
     -0.04738694801926613,
     -0.08439476788043976,
     0.0415908508002758,
     -0.041572462767362595,
     -0.010933791287243366,
     0.09030492603778839,
     0.03581192344427109,
     0.0054101720452308655,
     -0.019616270437836647,
     -0.03635474666953087,
     -0.016711806878447533,
     0.0020370257552713156,
     -0.028113355860114098,
     -0.04126256704330444,
     -0.001804799190722406,
     0.04241916909813881,
     -0.021162457764148712,
     -0.02308211661875248,
     0.04195022210478783,
     -0.0007403767085634172,
     -0.029753640294075012,
     -0.011469571851193905,
     -0.06720540672540665,
     -0.07313007861375809,
     -0.03452838957309723,
     -0.011662526987493038,
     0.053241390734910965,
     -0.015091597102582455,
     -0.036030758172273636,
     -0.016887003555893898,
     -0.04404503107070923,
     0.014085964299738407,
     -0.03580527380108833,
     -0.06354424357414246,
     0.033078111708164215,
     -0.02525801584124565,
     -0.0056268637999892235,
     -0.11245778203010559,
     0.07925047725439072,
     0.05479782074689865,
     -0.0005858275108039379,
     -0.01501956395804882,
     -0.017695322632789612,
     0.0866142213344574,
     -0.08537580072879791,
     3.0844188294000226e-35,
     -0.09546462446451187,
     0.057269416749477386,
     -0.034642331302165985,
     0.08187614381313324,
     -0.0008287510718218982,
     0.022280050441622734,
     0.014986173249781132,
     0.07156039774417877,
     -0.07588417828083038,
     0.006777838803827763,
     0.02291259728372097,
     -0.017157617956399918,
     -0.06387332826852798,
     0.03295464068651199,
     0.014282693155109882,
     0.010454969480633736,
     0.09239634871482849,
     0.03981653228402138,
     0.025721460580825806,
     0.008685283362865448,
     -0.03218107298016548,
     0.12598223984241486,
     -0.05986469238996506,
     0.005342153832316399,
     -0.006248053628951311,
     0.04000422731041908,
     -0.07173663377761841,
     0.0837775245308876,
     0.03286336734890938,
     0.05679282918572426,
     0.024801673367619514,
     -0.0068328650668263435,
     -0.06536345183849335,
     -0.03953390568494797,
     -0.049159564077854156,
     0.05799853056669235,
     0.05245916545391083,
     0.0243538785725832,
     -0.08396613597869873,
     0.054944105446338654,
     0.08873391896486282,
     0.003828256856650114,
     -0.07243207097053528,
     -0.013224733993411064,
     0.06913841515779495,
     -0.04628103971481323,
     0.0404001884162426,
     0.0016797709977254272,
     0.07987002283334732,
     0.0657188817858696,
     0.034920308738946915,
     -0.007304141763597727,
     -0.00241685239598155,
     -0.1339014768600464,
     -0.011643750593066216,
     -0.03316346928477287,
     0.017375215888023376,
     -0.04090135544538498,
     -0.09049036353826523,
     -0.012458406388759613,
     0.006870772689580917,
     -0.013241132721304893,
     -0.008450180292129517,
     -0.08038722723722458,
     0.03798096254467964,
     0.0764351561665535,
     -0.036909669637680054,
     0.113162100315094,
     -0.02238430641591549,
     0.017070671543478966,
     -0.0166767705231905,
     0.03616037219762802,
     -0.08240620791912079,
     -0.0686417743563652,
     -0.046417877078056335,
     0.012730148620903492,
     0.05642896145582199,
     0.01566169410943985,
     -0.07125354558229446,
     -0.016848411411046982,
     0.03501853346824646,
     -0.015701744705438614,
     0.03198131546378136,
     0.05693328008055687,
     0.013958173803985119,
     0.07488507032394409,
     0.033532027155160904,
     -0.0014691473916172981,
     0.023315923288464546,
     -0.05462191626429558,
     -0.031147969886660576,
     0.02842565067112446,
     0.07109032571315765,
     0.0037089528050273657,
     -0.03453952074050903,
     -1.5640383210779873e-08,
     -0.06566974520683289,
     -0.01206175610423088,
     0.05956012383103371,
     -0.0192488394677639,
     -0.003991483710706234,
     -0.04495304077863693,
     -0.10237562656402588,
     0.027931291610002518,
     -0.05669461935758591,
     0.03110659122467041,
     -0.018854495137929916,
     -0.01588922180235386,
     -0.02505580335855484,
     0.002297294093295932,
     0.07609699666500092,
     0.0707290917634964,
     0.05034541338682175,
     -0.02338407188653946,
     -0.05328081175684929,
     -0.00801961962133646,
     0.06464149802923203,
     0.019566934555768967,
     0.0427999421954155,
     0.03563281521201134,
     -0.0791451707482338,
     -0.012026646174490452,
     0.00855748075991869,
     -0.012865195982158184,
     0.019606351852416992,
     0.011781282722949982,
     -0.03582940250635147,
     0.0775003433227539,
     0.038879744708538055,
     -0.004179816227406263,
     -0.03889528661966324,
     0.02564876340329647,
     0.06510782986879349,
     -0.009892801754176617,
     0.034304432570934296,
     0.0016276468522846699,
     -0.0778566375374794,
     -0.02541779913008213,
     0.07304561883211136,
     0.036292463541030884,
     -0.05064752325415611,
     0.06744565069675446,
     0.01680132932960987,
     -0.05424441397190094,
     0.02875041402876377,
     -0.000517979555297643,
     0.0164160318672657,
     0.024103602394461632,
     0.03936567157506943,
     0.004738398361951113,
     0.054749745875597,
     -0.01327397208660841,
     -0.011988054029643536,
     -0.058534856885671616,
     -0.10339979827404022,
     0.0874004140496254,
     0.05176525563001633,
     -0.02117198519408703,
     -0.00611020578071475,
     0.015159721486270428]



## Initialize Chromadb _Vector Store_ and **Store the Chunks** in vector representation


```python
## Create a chromadb vector Store

persistentdirector = "../chroma_db"

## initialize Chromadb with HuggingFace embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory = persistentdirector,
    collection_name = "rag_collection"
)

print(f"Vector store created with {vectorstore._collection.count()} vector")
print(f"Persisted to: {persistentdirector}")
```

    Vector store created with 16 vector
    Persisted to: ../chroma_db
    

### Test Similarity Search


```python
query ="What are the types of machine learning?"

# similar_docs = vectorstore.similarity_search(query, k=3)
similar_docs
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 4
          1 query ="What are the types of machine learning?"
          3 # similar_docs = vectorstore.similarity_search(query, k=3)
    ----> 4 similar_docs
    

    NameError: name 'similar_docs' is not defined


## Test Similarity with Scores 


```python
result_score = vectorstore.similarity_search_with_score(query, k=2)
result_score
```




    [(Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpdgdn75iy\\doc_0.txt'}, page_content='Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.'),
      0.5967955589294434),
     (Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_0.txt'}, page_content='Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.'),
      0.5967955589294434)]



## Augmenting the RAG with LLM.


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=500
)

```


```python
test_resp = llm.invoke("What is Retrival Augmented Generation (RAG) and how was it being used before discovering the usecase with LLM?")
test_resp
```




    AIMessage(content='Retrieval Augmented Generation (RAG) is a model that combines the strengths of retrieval-based and generation-based approaches in natural language processing. It uses a retriever to search for relevant information from a large corpus of text and then generates a response based on that information.\n\nBefore discovering the use case with Large Language Models (LLM), RAG was primarily used in question-answering systems and information retrieval tasks. It was used to improve the accuracy and relevance of responses by incorporating information from external sources. RAG was also used in chatbots and virtual assistants to provide more informative and contextually relevant responses to user queries.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 127, 'prompt_tokens': 34, 'total_tokens': 161, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-DMInxY4Hl7fO9lajlDnAGPffiSdnW', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d1707-ab87-78d3-87d3-5dac992bf9b5-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 34, 'output_tokens': 127, 'total_tokens': 161, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})




```python
test_resp.response_metadata
"""
{'token_usage': {'completion_tokens': 127,
  'prompt_tokens': 34,
  'total_tokens': 161,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'model_provider': 'openai',
 'model_name': 'gpt-3.5-turbo-0125',
 'system_fingerprint': None,
 'id': 'chatcmpl-DMInxY4Hl7fO9lajlDnAGPffiSdnW',
 'service_tier': 'default',
 'finish_reason': 'stop',
 'logprobs': None}
"""
test_resp.response_metadata['token_usage']['total_tokens']
```




    161



## Initialize LLM, RAG Chain, Prompt Template, Query the RAG System.

### Modern RAG Chain


```python
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```


```python
## convert vector Store to retriever

retriever = vectorstore.as_retriever(
    search_kwarg={"k":3} ## Retrieve top 3 relevant chunks
)
retriever
```




    VectorStoreRetriever(tags=['Chroma', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.chroma.Chroma object at 0x00000232BC7CDD30>, search_kwargs={})




```python
## Create prompt template
system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to aswer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

prompt
```




    ChatPromptTemplate(input_variables=['context', 'input'], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks.\nUse the following pieces of retrieved context to aswer the question.\nIf you don't know the answer, just say that you don't know.\nUse three sentences maximum and keep the answer concise.\n\nContext: {context}"), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={})])




```python
## Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)
document_chain
```




    RunnableBinding(bound=RunnableBinding(bound=RunnableAssign(mapper={
      context: RunnableLambda(format_docs)
    }), kwargs={}, config={'run_name': 'format_inputs'}, config_factories=[])
    | ChatPromptTemplate(input_variables=['context', 'input'], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks.\nUse the following pieces of retrieved context to aswer the question.\nIf you don't know the answer, just say that you don't know.\nUse three sentences maximum and keep the answer concise.\n\nContext: {context}"), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={})])
    | ChatOpenAI(profile={'max_input_tokens': 16385, 'max_output_tokens': 4096, 'text_inputs': True, 'image_inputs': False, 'audio_inputs': False, 'video_inputs': False, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': False, 'tool_calling': False, 'structured_output': False, 'image_url_inputs': False, 'pdf_inputs': False, 'pdf_tool_message': False, 'image_tool_message': False, 'tool_choice': True}, client=<openai.resources.chat.completions.completions.Completions object at 0x00000232C482EBA0>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x00000232C482F620>, root_client=<openai.OpenAI object at 0x00000232C482C050>, root_async_client=<openai.AsyncOpenAI object at 0x00000232C482F380>, temperature=0.2, model_kwargs={}, openai_api_key=SecretStr('**********'), stream_usage=True, max_tokens=500)
    | StrOutputParser(), kwargs={}, config={'run_name': 'stuff_documents_chain'}, config_factories=[])




```python
## Create The Final RAG Chain

rag_chain = create_retrieval_chain(retriever, document_chain)
rag_chain
```




    RunnableBinding(bound=RunnableAssign(mapper={
      context: RunnableBinding(bound=RunnableLambda(lambda x: x['input'])
               | VectorStoreRetriever(tags=['Chroma', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.chroma.Chroma object at 0x00000232BC7CDD30>, search_kwargs={}), kwargs={}, config={'run_name': 'retrieve_documents'}, config_factories=[])
    })
    | RunnableAssign(mapper={
        answer: RunnableBinding(bound=RunnableBinding(bound=RunnableAssign(mapper={
                  context: RunnableLambda(format_docs)
                }), kwargs={}, config={'run_name': 'format_inputs'}, config_factories=[])
                | ChatPromptTemplate(input_variables=['context', 'input'], input_types={}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks.\nUse the following pieces of retrieved context to aswer the question.\nIf you don't know the answer, just say that you don't know.\nUse three sentences maximum and keep the answer concise.\n\nContext: {context}"), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={})])
                | ChatOpenAI(profile={'max_input_tokens': 16385, 'max_output_tokens': 4096, 'text_inputs': True, 'image_inputs': False, 'audio_inputs': False, 'video_inputs': False, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': False, 'tool_calling': False, 'structured_output': False, 'image_url_inputs': False, 'pdf_inputs': False, 'pdf_tool_message': False, 'image_tool_message': False, 'tool_choice': True}, client=<openai.resources.chat.completions.completions.Completions object at 0x00000232C482EBA0>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x00000232C482F620>, root_client=<openai.OpenAI object at 0x00000232C482C050>, root_async_client=<openai.AsyncOpenAI object at 0x00000232C482F380>, temperature=0.2, model_kwargs={}, openai_api_key=SecretStr('**********'), stream_usage=True, max_tokens=500)
                | StrOutputParser(), kwargs={}, config={'run_name': 'stuff_documents_chain'}, config_factories=[])
      }), kwargs={}, config={'run_name': 'retrieval_chain'}, config_factories=[])




```python
response = rag_chain.invoke({"input": "What are LLMs?"})
```


```python
response
```




    {'input': 'What are LLMs?',
     'context': [Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpdgdn75iy\\doc_2.txt'}, page_content='Core Functionality: LLMs work by predicting the next token (word or subword) in a sequence based on context. This enables them to perform tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs such as images or audio when extended into Large Multimodal Models (LMMs)'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Core Functionality: LLMs work by predicting the next token (word or subword) in a sequence based on context. This enables them to perform tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs such as images or audio when extended into Large Multimodal Models (LMMs)'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpdgdn75iy\\doc_2.txt'}, page_content='Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text .'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text .')],
     'answer': 'Large Language Models (LLMs) are advanced deep learning systems trained on massive datasets—often billions or trillions of words—to understand, generate, and manipulate human-like language. They work by predicting the next token in a sequence based on context, enabling tasks like text generation, summarization, translation, code writing, sentiment analysis, and even multimodal outputs. LLMs are built on transformer architectures that use self-attention mechanisms to process sequences in parallel, capturing context and relationships between words even when far apart in text.'}



## Add new Documents to Existing Vector Store


```python
vectorstore
```




    <langchain_community.vectorstores.chroma.Chroma at 0x232bc7cdd30>




```python
new_document = """Reinforcement Learning

Reinforcement Learning is a self-learning algorithm that includes Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGO), robotics, and autonomous systems.
"""
```


```python
new_doc = Document(
    page_content=new_document,
    metadata={"source": "mannual_addition", "topic":"reinforcement_Learning"}
)
```


```python
new_doc
```




    Document(metadata={'source': 'mannual_addition', 'topic': 'reinforcement_Learning'}, page_content='Reinforcement Learning\n\nReinforcement Learning is a self-learning algorithm that includes Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGO), robotics, and autonomous systems.\n')




```python
## Add new docs to vector store

### Split the docs
new_chunks = text_splitter.split_documents([new_doc])

vectorstore.add_documents(new_chunks)
new_chunks, vectorstore
```




    ([Document(metadata={'source': 'mannual_addition', 'topic': 'reinforcement_Learning'}, page_content='Reinforcement Learning\n\nReinforcement Learning is a self-learning algorithm that includes Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGO), robotics, and autonomous systems.')],
     <langchain_community.vectorstores.chroma.Chroma at 0x232bc7cdd30>)




```python
print(f"Added {len(new_chunks)} new chunks to the vector store")
print(f"Total vectors now: {vectorstore._collection.count()}")
```

    Added 1 new chunks to the vector store
    Total vectors now: 17
    

## Advanced RAG Technique - Conversational Memory


```python
from langchain_classic.chains import create_history_aware_retriever # Makes the retriever understand the conversational context
from langchain_core.prompts import MessagesPlaceholder # placeholder for chat history in prompts
from langchain_core.messages import AIMessage, HumanMessage # Structuring the message
```


```python
## create prompt that includes the chat history
contextualized_q_system_prompt = """Given a chat hostory and the latest user question
which might refrence context in the chat history, formulate a standalone question 
whaich can be understood without the chat history. DO NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""


contextualized_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualized_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```


```python
## history aware retirval
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualized_q_prompt
)
history_aware_retriever
```




    RunnableBinding(bound=RunnableBranch(branches=[(RunnableLambda(lambda x: not x.get('chat_history', False)), RunnableLambda(lambda x: x['input'])
    | VectorStoreRetriever(tags=['Chroma', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.chroma.Chroma object at 0x00000232BC7CDD30>, search_kwargs={}))], default=ChatPromptTemplate(input_variables=['chat_history', 'input'], input_types={'chat_history': list[typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, Tag(tag='ai')], typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x00000232F1F7C360>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]]}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='Given a chat hostory and the latest user question\nwhich might refrence context in the chat history, formulate a standalone question \nwhaich can be understood without the chat history. DO NOT answer the question, \njust reformulate it if needed and otherwise return it as is.'), additional_kwargs={}), MessagesPlaceholder(variable_name='chat_history'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={})])
    | ChatOpenAI(profile={'max_input_tokens': 16385, 'max_output_tokens': 4096, 'text_inputs': True, 'image_inputs': False, 'audio_inputs': False, 'video_inputs': False, 'text_outputs': True, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False, 'reasoning_output': False, 'tool_calling': False, 'structured_output': False, 'image_url_inputs': False, 'pdf_inputs': False, 'pdf_tool_message': False, 'image_tool_message': False, 'tool_choice': True}, client=<openai.resources.chat.completions.completions.Completions object at 0x00000232C482EBA0>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x00000232C482F620>, root_client=<openai.OpenAI object at 0x00000232C482C050>, root_async_client=<openai.AsyncOpenAI object at 0x00000232C482F380>, temperature=0.2, model_kwargs={}, openai_api_key=SecretStr('**********'), stream_usage=True, max_tokens=500)
    | StrOutputParser()
    | VectorStoreRetriever(tags=['Chroma', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.chroma.Chroma object at 0x00000232BC7CDD30>, search_kwargs={})), kwargs={}, config={'run_name': 'chat_retriever_chain'}, config_factories=[])




```python
# create new document chain with history
qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved contex to answer the question.
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# create conversational RAG pipeline
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

print("Conversation RAG Chain created")

```

    Conversation RAG Chain created
    


```python
chat_history = []
# First question
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is machine learning?"
})
print("Q. What is machine learning?")
print(f"A. {result1['answer']}")
```

    Q. What is machine learning?
    A. Machine Learning is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programming. It revolves around three key components — Task (T), Experience (E), and Performance (P) — where models improve performance on a task through experience over time.
    


```python
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its main types?"
})
print("Q. What are its main types?")
print(f"A. {result1['answer']}")
```

    Q. What are its main types?
    A. Machine Learning is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without explicit programming. It revolves around three key components — Task (T), Experience (E), and Performance (P) — where models improve performance on a task through experience over time.
    


```python
result2
```




    {'chat_history': [],
     'input': 'What are its main types?',
     'context': [Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpdgdn75iy\\doc_0.txt'}, page_content='Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_0.txt'}, page_content='Core Types of ML include:\n    Supervised Learning – Uses labeled data for tasks like classification (predicting categories) and regression (predicting continuous values).\n    Unsupervised Learning – Works with unlabeled data to find patterns, such as clustering or dimensionality reduction.\n    Reinforcement Learning – Learns optimal actions through trial and error to maximize rewards.\n    Additional: Semi-Supervised and Self-Supervised Learning for scenarios with limited labeled data.'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpdgdn75iy\\doc_2.txt'}, page_content='Large Language Models'),
      Document(metadata={'source': 'C:\\Users\\NILANJ~1.PAU\\AppData\\Local\\Temp\\tmpgi6jnjye\\doc_2.txt'}, page_content='Large Language Models')],
     'answer': 'The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Additional types include semi-supervised and self-supervised learning.'}


