# VanGogh RAG Model

A RAG (Retrieval-Augmented Generation) model is a hybrid approach in natural language processing (NLP) that combines retrieval and generation techniques. It is particularly effective in tasks like open-domain question answering, summarization, and information retrieval. Here's an overview:

How RAG Works

data pre-processing: 
First of all, the data is recursively split in user-defined chuncks, then embedded using an embedding model, then the embedded data is stored as vectors. Lastly, a retriever is able to extract the vectors (read: chunks of text) are most appropriate to provide an answer to the user query. 

Retrieval Component:
The model searches for relevant pieces of information (documents, paragraphs, or facts) from an external knowledge base or corpus.
A retriever (e.g., a dense passage retriever or vector-based search) identifies and ranks the most relevant documents based on a query.

Augmentation:
The retrieved information is used to enrich the input context for the generative component.
The goal is to provide the generator with supporting evidence or context to improve its response.

Generation Component:
The generative model (usually a large language model like GPT or BERT) processes the query alongside the retrieved context.
It generates a response or completes a task using the combined query and supporting data.

- **Knowledge-Enhanced Generation**: RAG models augment their understanding by incorporating external, often up-to-date, knowledge, reducing the need for the model itself to "memorize" all information.

- **Improved Accuracy**: By grounding its generation in retrieved evidence, the model reduces hallucination (i.e., generating plausible but incorrect information).

- **Dynamic Knowledge Integration**: Unlike static pre-trained models, RAG can use external sources (like databases or live APIs), allowing it to work with up-to-date or domain-specific data.

The RAG_local_model file was used to experiment with running the RAG model locally in the IDE. Information is in the ipynb file. In order to run the flask application with a (very simple, just for prototype purposes) user interface, make a virtual environment in your terminal. First, clone the repo to a local directory and open it in your IDE. In the virtual environment, install the requirements.txt file and run the app.py file. A URL is created in the terminal, which provides the opportunity to experiment with the user interface interactively. The script.js file in the static folder, and the index.html file in the templates folder configure the User Interface. Leave them in the same folders. 

In order to get this to production, it should be containerized using docker (as example) and run on hardware resources that decrease latency, or be run in the cloud. The project should also be fined tuned a lot still. Fine-tuning can simply be done by adjusting the code in either the app.py file, script.js file, or index.html. Remember that changes in 1 file often also create a need for change to the other 2 files. 
