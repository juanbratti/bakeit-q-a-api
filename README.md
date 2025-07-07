# Prueba Técnica Bircle.AI

# BakeitQ&A

Como solución a la prueba técnica, lo que desarrollé fue una API utilizando FastAPI que responde preguntas sobre un corpus de recetas de repostería. Las recetas se dividen por categoría de postre (un txt para cada categoría).

Utiliza:

- ChromaDB como vector store para RAG.
- OpenAIEmbeddings para realizar los respectivos embeddings.
- LlamaIndex para la implementación en general.
- Chunking manual y reranking.

**Como se pide en la consigna, la estructura es la siguiente:**

```markdown
requirements.txt
app/
■■ .env # agregar manualmente
■■ main.py # FastAPI app y endpoints
■■ rag.py # Lógica de RAG con Chroma
■■ orchestrator.py # (Opcional) Workflow con agente LlamaIndex
■■ data/ # Documentos (.txt) a indexar
■■■■■ cake_recipes.txt
■■■■■ cookies_recipes.txt
■■■■■ frozen_recipes.txt
■■■■■ other_recipes.txt
■■■■■ pie_recipes.txt
■■■■■ pudding_recipes.txt
```

# Ejecución

Incluí el archivo `requirements.txt` especificando las librerías que usé.

Para correrlo:

1. Crear virtual environment `python -m venv venv` y activarlo con `source venv/bin/activate`.
2. Realizar `pip install -r requirements.txt` .
3. Crear archivo `app/.env` que incluya `OPENAI_API_KEY="..."` .
4. Pararse en  `app/` y ejecutar `fastapi run main.py`.

Al momento de ejecutar se van a ver unos logs que los utilicé a modo de debugging.

# Implementación

A continuación describo los datos y la implementación + decisiones.

## Datos

Cada archivo en `data` es una recopilación de recetas de la categoría especificada en el nombre del archivo. Por ejemplo, en el caso de `cake_recipes.txt` su contenido son recetas de tortas:

```markdown
Recipe: Chhena Poda (Spiced Cheesecake)
Category: cake
Complexity: 11 ingredients, 18 steps

Ingredients needed:
Ghee, for the cake pan, 8 ounces paneer, preferably homemade, ¼ cup confectioners’ sugar, ½ teaspoon ground cardamom, 2 tablespoons semolina, 1½ tablespoons ghee, store-bought or homemade, ¼ cup whole milk, 2 tablespoons roasted cashews, 2 tablespoons golden raisins, 3 tablespoons granulated sugar, Freshly whipped cream or sliced almonds (optional), for serving

Cooking instructions:
Preheat the oven to 350°F. Butter a 9-inch nonstick metal cake pan with some ghee. In a large bowl, stir together the paneer, confectioners’ sugar, and cardamom until incorporated but still slightly chunky. Add the semolina, ghee, and milk and whisk until smooth. Stir in the cashews and raisins and set aside. Sprinkle the granulated sugar over the bottom of the prepared cake pan. Set the pan on the stove over medium heat and once the sugar just begins to bubble and turn pale golden brown (the sugar will continue to caramelize in the oven, so don’t let it get any darker right now), after 8 to 10 minutes, remove the pan from the heat using tongs. (This step is a bit tricky and you might be tempted to move the sugar around with a spoon while it is bubbling, but don’t touch it—just let it caramelize on its own.) You will know it’s ready for the cake batter when it’s bubbling but is still slightly grainy and just turning a pale golden color. Pour the batter into the pan, using a rubber spatula to scrape the bowl clean. It should settle into an even layer on its own, but if it doesn’t, use the spatula to create a smooth surface. Wearing oven mitts, carefully transfer the pan to the oven and bake until a toothpick inserted into the center of the cake comes out clean, 40 to 45 minutes. Transfer to a wire rack and cool in the pan to room temperature. Run a paring knife around the cake’s edges and then place a serving platter larger than the diameter of the cake over the cake and gently flip the cake over—it should slide out, with caramel coming out with it like you would see in a flan (though some caramel will be left in the pan—this is fine). Slice and serve with freshly whipped cream or sliced almonds, if desired. The cake will keep in a covered container in the refrigerator for up to 3 days. Note: To clean the pan, add some boiling water to the pan and let it sit for at least 30 minutes to loosen up the browned sugar sticking to the surface of the pan before cleaning it.

This is a cake recipe with 11 ingredients.

==================================================

Recipe: Chocolate Zucchini Cake
Category: cake
Complexity: 13 ingredients, 13 steps

Ingredients needed:
2 1/4 cups sifted all purpose flour, 1/2 cup unsweetened cocoa powder, 1 teaspoon baking soda, 1 teaspoon salt, 1 3/4 cups sugar, 1/2 cup (1 stick) unsalted butter, room temperature, 1/2 cup vegetable oil, 2 large eggs, 1 teaspoon vanilla extract, 1/2 cup buttermilk, 2 cups grated unpeeled zucchini (about 2 1/2 medium), 1 6-ounce package (about 1 cup) semisweet chocolate chips, 3/4 cup chopped walnuts

Cooking instructions:
Preheat oven to 325°F. Butter and flour 13 x 9 x 2-inch baking pan. Sift flour, cocoa powder, baking soda and salt into medium bowl. Beat sugar, butter and oil in large bowl until well blended. Add eggs 1 at a time, beating well after each addition. Beat in vanilla extract. Mix in dry ingredients alternately with buttermilk in 3 additions each. Mix in grated zucchini. Pour batter into prepared pan. Sprinkle chocolate chips and nuts over. Bake cake until tester inserted into center comes out clean, about 50 minutes. Cool cake completely in pan.

This is a cake recipe with 13 ingredients.

==================================================
...
```

Descripción general de los datos:

| **Archivo** | **Cantidad** | **Categoría** |
| --- | --- | --- |
| cake_recipes.txt | 35 recetas | Pasteles y tortas |
| cookies_recipes.txt | 25 recetas | Galletas |
| frozen_recipes.txt | 5 recetas | Postres helados |
| other_recipes.txt | 47 recetas | Otros postres |
| pie_recipes.txt | 26 recetas | Tartas y pies |
| pudding_recipes.txt | 2 recetas | Pudines |

Debido a que cada receta esta delimitada por una línea que contiene ‘`====…`’, aproveché ese detalle para realizar chunking de forma manual, teniendo 1 chunk por receta. Por qué hice esto?

1. Las recetas no son muy largas. Aproximadamente cada una es equivalente a ~500 tokens (usé [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) para verificar).
2. La idea de la API es devolver recetas a modo de sugerencia, por lo que las recetas son las unidades de información que quiero manejar.
3. Hacer chunks con mayor cantidad de tokens no solo hace que el vector que hace el embedding tenga mayor ruido, si no que también al pasar los documentos al modelo de lenguaje podemos tener:
    - Recetas con información cortada (pérdida de información) → se traduce a peor respuesta del modelo de lenguaje.
    - Pedazos de recetas que son relevantes a la consulta + pedazos de recetas que no tienen nada que ver → se traduce en mayor cantidad de tokens usados en la API por lo tanto más $$$.

Entonces, no usé `SimpleDirectoryReader` de LlamaIndex para obtener los documentos de tipo `Document` que se les pasa al `VectorStoreIndex`, si no que definí mis propios documentos (de tipo `Document`) parseando las recetas con una función específica `_load_recipes_from_txt`. Explico esto en más detalle en el **módulo de rag.py**.

PD: (los datos los obtuve del dataset https://im2recipe.csail.mit.edu/.)

## Módulo main.py

Acá se tiene la implementación de los endpoints pedidos `GET /query?q=` y `GET /agent_query?q=`. Cada endpoint tiene un modelo de respuesta definido con Pydantic + la inicialización del agente orquestador y del sistema RAG.

### Modelos de Pydantic

Para el endpoint `/query` tenemos el siguiente modelo:

```python
class QueryResponse(BaseModel): 
	  # las 3 cosas que se piden
    query: str
    documents: List[str]
    response: str
```

y para el endpoint `/query_agent` tenemos el siguiente modelo:

```python
class AgentResponse(BaseModel):
    response: str
```

ambos sirven para la validación y serialización en JSON de las respuestas de los endpoints.

`AgentResponse` tiene un único campo `response` porque es lo único que se devuelve al hacer run sobre el `FunctionAgent` de LlamaIndex.

### Endpoint /query?q=

Responde queries del usuario buscando en los documentos indexados (recetas) usando el enfoque RAG. La respuesta tiene que cumplir el formato de la `QueryResponse`. Se realiza manejo de errores en caso de excepciones.

```python
# Initialize RAG system
from rag import RAG
rag = RAG()

@app.get("/query", response_model=QueryResponse)
async def query_recipes(q: str = Query(..., description="Search query for recipes using RAG")):
    """Query the RAG system for recipes"""
    try:
        response = rag.query(q)
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Endpoint /query_agent?q=

Responde preguntas del usuario pero usando el agente orquestador de LlamaIndex, que sigue un workflow más estructurado. Se realiza manejo de errores en caso de excepciones.

```python
# Initialize Agent Orchestrator
from orchestrator import AgentOrchestrator
agent_orchestrator = AgentOrchestrator()

@app.get("/agent_query", response_model=AgentResponse)
async def agent_query_recipes(q: str = Query(..., description="Search query for recipes using the Agent Orchestrator")):
    """Query the Agent Orchestrator for recipes, with planning and reasoning"""
    try:
        response = await agent_orchestrator.run(q)
        return AgentResponse(response=response)
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Las validaciones sobre como es la consulta las hago dentro de las funciones que llaman los endpoints (`agent_orchestrator.run` y `rag.query(q)`)

## Módulo rag.py

Para el desarrollo de este módulo, me base en la documentación de LlamaIndex: [https://docs.llamaindex.ai/en/stable/understanding/rag/](https://docs.llamaindex.ai/en/stable/understanding/rag/).

Para la implementación del sistema RAG, decidí basarme en un diseño orientado a objetos. Esto es porque me parece mucho más modularizable y adaptable el sistema a cambios. 

Intenté modularizar lo más que pude para limpiar la implementación del método que hace todo el trabajo (`query`)

### \_\_init\_\_

Este método es para inicializar el sistema Sus argumentos son hiperparámetros para probar distintas configuraciones. 

Tiene dos comportamientos posibles dependiendo de si es una inicialización nueva o si ya se inicializó el sistema anteriormente. 

Los argumentos son los siguientes:

```python
def __init__(self,      # I'd rather have a config file for these values
                 chroma_path: str = "./chromadb",
                 data_path: str = "./data",
                 collection_name: str = "dessert_recipes",
                 llm_model: str = "gpt-3.5-turbo", # cohere: "command-r-plus-04-2024"
                 temperature: float = 0.5,
                 embedding_model: str = "text-embedding-3-small", # cohere: "embed-english-v3.0",
                 top_k: int = 20,
                 top_n: int = 5, # this is for reranking
                 index_path: str = "./index_storage",
                 similarity_threshold: float = 0.3,
                 ):         # I keep them this way to follow the instructions of the project

```

A mi me parece un poco mas limpio sacar los parámetros y escribirlos en un archivo aparte, pero los dejé ahí para respetar la estructura pedida. Salteando los obvios:

- `llm_model`: use `gpt-3.5-turbo` porque es un modelo barato y de propósito general.
- `temperature`: nivel de creatividad por parte del modelo de lenguaje. La seteé en 0.5 porque es el recomendado para propósitos generales.
- `llm_embedding_model`: usé `text-embedding-3-small` también por una cuestión del costo-beneficio. Según la documentación de OpenAI ([https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)), entre los 3 modelos de embedding que tiene, el utilizado acá es el menos costoso, y está en 2do lugar según su score. El que está en primer lugar es text-embedding-3-large, pero la diferencia absoluta entre el score del mismo con el que usé es de 2.3% (no tanta). Por ende usé la versión small.
- `top_k:` el retrieval al hacer la búsqueda en la base de datos vectorial es de 20 documentos. Si bien puede parecer mucho, recordar que los chunks son más pequeños (1 por receta), y además no le voy a pasar los 20 documentos al LLM porque aplico reranking y filtro por similaridad (explico abajo).
- `top_n`: decidí utilizar reranking para probar el retrieval de documentos. Este es el parámetro que indica cuántos documentos devolver al hacer reranking (es decir, de 20, se sacan los mejores 5). Lo seteé en 5 porque el retrieval de la vector db es bastante bueno tomando 1 chunk por receta, asique basta con una ventana de 5 chunks para responder una query.
- `similarity_threshold`: lo usé para que (antes de hacer reranking) se descarten los documentos que NADA tienen que ver con la query (por eso el threshold bajo de 0.3), para evitar rerankear documentos que de igual manera no sirven para responder la query. Disminuye la cantidad de tokens al generar respuesta y ruido.
- `chroma_path` e `index_path` son para aprovechar la persistencia de ambas cosas, y no tener que recrear, recargar y reindexar las bases de datos. Una vez que se inicializan en este método, se reutilizan sus instancias persistentes que se encuentran en los paths indicados acá.

Después, la implementación se divide en los siguientes pasos:

1. Validaciones, chequeo de los directorios + guardo hiperparámetros que reuso en otros métodos:
    
    ```python
    # basic validations, just in case
    if not os.path.exists(data_path):
    	raise ValueError(f"Data not found: {data_path}")
    if not collection_name.strip():
    	raise ValueError("Not a valid collection name")
    if top_k <= 0 or not (0 <= temperature <= 1):
      raise ValueError("Invalid top_k or temperature")
    
    # this is for later use in other methods of the class
    self.data_path = data_path
    self.index_path = index_path
    self.top_k = top_k
    self.top_n = top_n
    self.similarity_threshold = similarity_threshold
    
    # makes sure the paths to the chroma db & index exist
    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)
    ```
    
2. Inicialización del cliente de chroma y guardado de la chroma store vector, configuración del modelo de embedding y lenguaje usando `Settings` de `LlamaIndex` .
    
    ```python
    # obtains the chroma client
    self.chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    # gets the chroma collection or creates it if it doesn't exist
    self.collection = self.chroma_client.get_or_create_collection(collection_name)
    
    # I use the Settings class from LlamaIndex to set the embedding & llm models.
    # this is just one layer more of abstraction to make it easier to change the models if needed.
    Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    Settings.llm = OpenAI(model=llm_model, temperature=temperature, api_key=openai_api_key)
    
    # sets the vector store with the corresponding chroma collection
    self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
    ```
    
    Acá abajo también cambio la configuración del `node_parser` de LlamaIndex. Esto es porque si bien hago chunking manual definiendo mi propia lista de elementos tipo `Document`, al guardar las recetas en el índice de LlamaIndex usando `VectorStoreIndex` por primera vez, LlamaIndex aplica chunking igual sobre los `Document` definidos por mí (o sea, a cada receta la divide en más chunks). 
    
    Esto es porque al parecer tiene un chunk_size menor a ~500. Para evitar que haga chunking de nuevo, lo que hago es setearle el chunk_size en 1000, para que la ventana de tokens sea mayor que la cantidad de tokens por receta ~500), aceptando mis chunks tal como los definí yo (1 por receta):
    
    ```python
    # The following setting is to not have my custom chunks splitted by default
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=0)
    ```
    
3. Ahora sí, tenemos dos opciones según si es una inicialización por primera vez o no. Esto lo chequeo viendo si efectivamente existe un índice en el index_path (abstraigo la operación usando `_index_exists` 
    1. SI existe el index, lo recargo únicamente:
        
        ```python
        if self._index_exists():
                        # if the index is persisted and the chroma collection is not empty, loads it
                        self.storage_context = StorageContext.from_defaults(
                            vector_store=self.vector_store,
                            persist_dir=self.index_path,
                        )
        
                        # this loads the index from the persisted storage
                        self.index = load_index_from_storage(self.storage_context)
        ```
        
    2. caso contrario, lo creo:
        
        ```python
        else:
        # otherwise -> creates new storage context and ingests documents
                        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                        self._index_ingest()
        ```
        
        para crearlo llamo a `_index_ingest`.
        
4. Finalmente, como ultimo paso construyo y guardo el query engine para reutilizarlo en todas las queries hechas.  Explico como hago el query_engine esto más abajo.
    
    ```python
    self.query_engine = self._build_query_engine()
    ```
    

### _index_exists

Es simplemente para chequear si el índice existe. Chequeo que no esté vacío el path donde estaría el índice, y además que la colección no sea cero (si es cero, el índice no se inicializó).

```python
def _index_exists(self) -> bool:
	"""Checks if the index is persisted and collection is not empty"""
	# if the index is empty -> listdir is an empty list -> bool([]) = False
	# I also use the collection.count() just in case.
	return bool(os.listdir(self.index_path)) and self.collection.count() > 0
```

### _index_ingest

Se usa únicamente cuando se inicializa por primera vez. Obtiene la lista de chunks que hice manualmente (`recipes_docs`), chequea que no sea vacía (si no, el txt sería vacío) y luego ingesta los documentos en la vector store, usando el `storage_context` respectivo y haciéndolo persistente para reutilizarlo después.

```python
def _index_ingest(self) -> None:
  """Ingest the documents into the index. Assumes the data path is valid."""
  # loads the documents from the data path
  recipes_docs = self._load_recipes_from_txt()

  # if the txt file is empty, raises an error
  if not recipes_docs: 
      raise RuntimeError("Please provide recipes :(")
  
  # otherwise -> ingests the documents into the vector store
  self.index = VectorStoreIndex.from_documents(
      recipes_docs,
      storage_context=self.storage_context,
  )

  # the index is persisted in the index_path for later re-use
  self.index.storage_context.persist(persist_dir=self.index_path)
```

### _load_recipes_from_txt

La usa _index_ingest para obtener los chunks manuales. Es específico para mis datos. En caso de querer hacerlo funcionar para cualquier tipo de txt file, hay que usar el `SimpleDirectoryReader` de LlamaIndex para que haga chunking automático.

```python
def _load_recipes_from_txt(self) -> List[Document]:
	"""This is for the preprocess of the txt file. Loads recipes from a text file 
	where each recipe is separated by a line of ==='s.
	This custom loader is only used for the specific data I found on the internet.
	It won't work with arbitrary txt files.
	"""
	
	docs = []
	for file in os.listdir(self.data_path):
		file_path = os.path.join(self.data_path, file)
		with open(file_path, "r", encoding="utf-8") as f:
			text = f.read()
		recipes = re.split(r"\n={5,}\n", text)
		recipes_list = [recipe.strip() for recipe in recipes if recipe.strip()]
		docs.extend([Document(text=recipe) for recipe in recipes_list]) 
		# I don't add metadata because it adds more complexity to the code.
		
	return docs
```

Como cada receta tiene su categoría explícita en su información, podría haber considerado agregar metadata a cada chunk y usar esa metadata para hacer mejor retrieval ante queries, decidí no hacerlo porque no son muchas las recetas y no me parecía que iba a traer demasiado beneficio con las cosas que ya se usan (reranking, threshold de similitud).

### _translate_query

La siguiente es una función que uso en `query` (la que procesa la consulta). Es básicamente para traducir la consulta, porque OpenAI no tiene un modelo de embedding multilingual, entonces como las recetas están en ingles (y las guardo así en chroma), conviene que la consulta también esté en inglés para que haya mayor correspondencia semántica al buscar recetas en chroma. 

Usa una llamada a la API de OpenAI para traducir, porque no es traducción usando reglas, ni heurísticas.

```python
def _translate_query(self, query: str) -> str:
	"""In case the given query is NOT in english, this method translates it.
    This maintains consistency and allows for better retrieval of the recipes."""
    
    try:
        translation_prompt = f"""
        Translate the following query into English as accurately as possible.
        Return ONLY the translated text, no explanations, no additional commentary.
        Preserve the original meaning exactly, without adding, removing, or altering 
        any information.

    Query: {query}

    English translation:
    """
    translation = Settings.llm.complete(translation_prompt)
    result = str(translation).strip()
        
    if not result:
        raise RuntimeError("Translation returned empty result")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}")
```

### _is_likely_english

Esta función es para evitarme la llama de la API de OpenAI para traducir la consulta (en el caso de que la consulta ya esté en inglés). Usa la librería `langdetect` para ver si el lenguaje de la consulta es inglés o no. Si ya es inglés, no se va a hacer la traducción con `_translate_query`.

```python
def _is_likely_english(self, text: str) -> bool:
	"""Simple heuristic to check if text is likely English. 
		This saves a call to the model's api."""
	
	return detect(text) == 'en'
```

### _get_prompt_template

Esta función es únicamente para limpiar más la implementación de `query`. Uso `PromptTemplate` de `LlamaIndex` para simplificar el pasaje del contexto y la query (la hace LlamaIndex automáticamente al llamar al query engine).

Acá fui bastante explícito con las tareas del modelo de lenguaje. Me pasó que se inventaba recetas, o que no incluía los ingredientes. Por eso tuve que especificar varias de las cosas que tenía (o no) que hacer.

```python
def _get_prompt_template(self) -> PromptTemplate:
  """Returns my custom prompt template for the query.
  This is the prompt that the query engine will use to generate the response."""

  # custom prompt template
  return PromptTemplate(
        """You are a culinary assistant specialized in providing 
          recipes based on a given context.
          Your task is to respond to user queries using ONLY the 
          recipes available in the provided context.
          Your answer must be friendly and include the following 
          information:

          1. Context: You will be given a list of recipes, each with 
          its name, ingredients, and detailed instructions, including 
          exact quantities and temperatures.
          2. Query: You will receive a query that may include keywords 
          related to recipes or ingredients.

          Instructions:
          - Upon receiving the user's query, first search the names of 
          the recipes in the context for matches with the query's keywords.
          - If you find recipes with matching keywords in their names, 
          select those recipes and provide:
          1. Recipe Name
          2. Ingredients (including exact quantities)
          3. Instructions (step-by-step, including specific temperatures)
          - If no recipes match the keywords in the name, search for 
          recipes that contain ingredients mentioned in the query.
          For those recipes, also provide:
          1. Recipe Name
          2. Ingredients (including exact quantities)
          3. Instructions (step-by-step, including specific temperatures)

          - Ensure your answer is clear, concise, and friendly. Use a welcoming 
          and encouraging tone to motivate the user to try the recipes.

          Context:
          {{context_str}}

          Query:
          {{query_str}}
          """
  ) # just in case: context_str and query_str are passed internally by the query engine
```

### _build_query_engine

Esto construye el query engine y lo guarda para ser reutilizado ante otras consultas. 

Notar que construí un query engine de forma diferente porque podría haber usado la función `as_query_engine()` de LlamaIndex para simplemente hacer la query directamente. Decidí no usar eso porque hacer tu propio query engine te da mas libertades y te permite usar reranking, aplicar threshold de similaridad, hacer procesamiento específicos a los nodos, entre otras cosas. 

Entonces, para construir el query engine tuve que usar el `RetrieverQueryEngine` de LlamaIndex, definiendo así:

- El `retriever` usando `VectorIndexRetriever`. Este se encarga de la indexación y de devolver los k documentos.
- Un `response_synthesizer` usando `get_response_synthesizer` de LlamaIndex. Este sirve para armar la respuesta final con el prompt mío.
- El `reranker` usando `LLMRerank` de LlamaIndex para ordernar por relevancia los k documentos devueltos
- Y finalmente uno todos los componentes con el `RetrieverQueryEngine`


```python
def _build_query_engine(self) -> RetrieverQueryEngine:
	"""Builds the query engine with the retriever and the custom prompt.
    It uses the VectorIndexRetriever because I want to toggle the similarity threshold."""
    
	# configuration of the retriever
    retriever = VectorIndexRetriever(
        index=self.index,
        similarity_top_k=self.top_k,
    )

	# configuration of the response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", # documentation says it does less api calls 
        text_qa_template=self._get_prompt_template(), # usage of my custom prompt
    )

	# I apply reranking to get the top_n most relevant results
	reranker = LLMRerank(
		top_n=self.top_n,
		llm=Settings.llm,
		)

	# this is the query engine that will be used to query the index
	return RetrieverQueryEngine(
		retriever=retriever,
		response_synthesizer=response_synthesizer,
		# I use the similarity threshold to filter out extreme non-relevant results + reranker
		node_postprocessors=[
			SimilarityPostprocessor(similarity_cutoff=self.similarity_threshold),
			reranker,
		    ],
    )
    # I could've used the reranker only, but sometimes (in random queries) 
    # some docs were retrieved anyway and passed to the LLM, which 
    # was a waste of tokens.
```

### _query_engine y _get_documents

- El _query_engine es para abstraer la llamada al query engine construído
- El _get_documents es para abstraer la extracción de los documentos de la respuesta del query engine.

```python
def _query_engine(self, english_query: str) -> Response: 
	"""Returns the results of the query engine.
	This doesn't return the documents on their own, but rather the source nodes."""
	
	results = self.query_engine.query(english_query)
	return results
    
def _get_documents(self, results: Response) -> List[str]:
	"""Gets the documents from the results.
	This is used to return the documents on their own."""
	
	source_nodes = results.source_nodes
	documents = [node.node.get_content() for node in source_nodes]
	return documents
    
```

### query

Acá abstraigo la mayoría de las operaciones usando las funciones que ya expliqué arriba.

```python
def query(self, query: str) -> Dict:
	"""Main method to query the RAG system.
	This is the method that will be used to query the RAG system."""
	
	# basic validations
	original_query = query.strip()
	if not original_query:
		return {
			"query": query,
			"documents": [],
			"response": "Please provide a valid query."
		}
		
	try:            
		# translate the query to English for better retrieval
		if not self._is_likely_english(original_query):
			english_query = self._translate_query(original_query)
		else:
			# If query is very likely already in English, skip translation
			english_query = original_query
			
		# gets documents
		results = self._query_engine(english_query)
		documents = self._get_documents(results)
		
		# CASE 1: no relevant docs found
		if not documents:
			return {
				"query": query,
				"documents": [],
				"response": "No recipes found for your query :(",
				}
    
        # CASE 2: relevant docs found            
        # the response synthesizer already handles the response generation by the llm
        response_text = str(results.response)
        return {
            "query": query,
            "documents": documents,
            "response": response_text,
            }
	except Exception as e:
		return {
			"query": query,
			"documents": [],
			"response": "Sorry, an unexpected error occurred. Please try again."
			}

```

## Ejemplo 1: Pregunta sobre tortas de chocolate

Query: What chocolate cakes can I bake?

JSON devuelto:

```json
{
  "query": "What chocolate cakes can I bake?",
  "documents": [
    "Recipe: Chocolatey Chocolate Cake\nCategory: cake\nComplexity: 12 ingredients, 17 steps\n\nIngredients needed:\n1 3/4 cups (225g) all-purpose flour, 1/2 cup (45g) unsweetened Dutch-processed cocoa powder, 1 1/2 teaspoons baking powder, 1 1/2 teaspoons baking soda, 1/2 teaspoon salt, 1 1/2 cups (300g) granulated sugar, 2 eggs, at room temperature, 1 cup (240ml) whole milk, 1/2 cup (120ml) grapeseed oil or any mild-flavored oil, 1/2 teaspoon pure vanilla extract, 1 cup (240ml) boiling water, Confectioners' sugar, chocolate Swiss meringue buttercream, warm ganache, marshmallow icing, whipped cream\n\nCooking instructions:\nPreheat the oven to 350°F. Grease a 10 by 3-inch round pan with butter, line the bottom and sides of the pan with parchment paper, and grease the paper. (I’ll let you just grease, line the bottom of the pan, and lightly flour the sides if you’re feeling lazy.) Place a large sifter or a sieve in a large mixing bowl. Add the flour, cocoa, baking powder, baking soda, and salt and sift. Add the sugar and whisk until combined. In another large bowl, whisk the eggs, milk, oil, and vanilla together. Gradually add the wet ingredients to the dry ingredients and whisk until there are no lumps and the batter is smooth. Carefully pour in the boiling water and stir until combined. (Watch the little ones with the hot water!) Pour the batter into the prepared pan. Bake in the center of the oven for approximately 50 minutes or until a wooden skewer inserted in the center comes out clean, and the cake bounces back when lightly pressed. Remove from the oven and let the cake stand for 10 minutes. Run a butter knife around the cake to gently release. Peel off the parchment paper from the sides. Invert the cake, peel off the bottom piece of parchment, and cool on a wire rack. Coffee Enhanced Chocolate Cake: Swap the cup of water for a cup of hot black coffee.\n\nThis is a cake recipe with 12 ingredients.",
    "Recipe: Chocolate Zucchini Cake\nCategory: cake\nComplexity: 13 ingredients, 13 steps\n\nIngredients needed:\n2 1/4 cups sifted all purpose flour, 1/2 cup unsweetened cocoa powder, 1 teaspoon baking soda, 1 teaspoon salt, 1 3/4 cups sugar, 1/2 cup (1 stick) unsalted butter, room temperature, 1/2 cup vegetable oil, 2 large eggs, 1 teaspoon vanilla extract, 1/2 cup buttermilk, 2 cups grated unpeeled zucchini (about 2 1/2 medium), 1 6-ounce package (about 1 cup) semisweet chocolate chips, 3/4 cup chopped walnuts\n\nCooking instructions:\nPreheat oven to 325°F. Butter and flour 13 x 9 x 2-inch baking pan. Sift flour, cocoa powder, baking soda and salt into medium bowl. Beat sugar, butter and oil in large bowl until well blended. Add eggs 1 at a time, beating well after each addition. Beat in vanilla extract. Mix in dry ingredients alternately with buttermilk in 3 additions each. Mix in grated zucchini. Pour batter into prepared pan. Sprinkle chocolate chips and nuts over. Bake cake until tester inserted into center comes out clean, about 50 minutes. Cool cake completely in pan.\n\nThis is a cake recipe with 13 ingredients.",
    "Recipe: Camouflage Chocolate Fudge Brownies\nCategory: other\nComplexity: 10 ingredients, 32 steps\n\nIngredients needed:\nNonstick cooking oil spray or room-temperature unsalted butter (for pan), 8 oz. cream cheese (not low-fat), cut into (1\") pieces, 3 large eggs, chilled, 1⅓ cups (266 g) sugar, divided, 1 tsp. vanilla extract, divided, ¾ tsp. kosher salt, divided, 1½ tsp. plus ¾ cup plus 2 Tbsp. cocoa powder, preferably Dutch-process, 10 Tbsp. unsalted butter, cut into pieces, 1 tsp. instant espresso powder (optional), ½ cup (63 g) all-purpose flour\n\nCooking instructions:\nPlace a rack in middle of oven; preheat to 325°F. Lightly coat a 9x9\" pan, preferably metal, with nonstick spray. Line with parchment paper, leaving overhang on all sides. Lightly coat parchment with nonstick spray. Place cream cheese in a medium heatproof bowl set over a medium saucepan of barely simmering water (do not let bowl touch water). Heat cream cheese, stirring occasionally, until very soft, about 5 minutes. Remove bowl from heat (leave water simmering). Using a heatproof rubber spatula or wooden spoon, smash and mix cream cheese until smooth. Add 1 egg, ⅓ cup (66 g) sugar, ½ tsp. vanilla, and ¼ tsp. salt and whisk until very smooth. Transfer about half of cream cheese mixture to a small bowl and whisk in 1½ tsp. cocoa powder. Place butter in another medium heatproof bowl. Add espresso powder (if using) and remaining 1 cup (200 g) sugar, ¾ cup plus 2 Tbsp. cocoa powder, and ½ tsp. salt. Place bowl over saucepan of still simmering water and cook, stirring occasionally once the butter starts to melt, until mixture is homogeneous and too hot to leave your finger in, 7–9 minutes. Let cool 5 minutes. Add remaining 2 eggs and remaining ½ tsp. vanilla to butter mixture one at a time, whisking vigorously after each addition until smooth and glossy. Add flour and mix with spatula or spoon until no longer visible, then vigorously mix another 30 strokes. Scoop out ½ cup brownie batter and set aside. Scrape remaining batter into prepared pan and spread into an even layer. Working quickly, alternate dollops of cocoa–cream cheese mixture and cream cheese mixture over batter. Dollop reserved batter on top (it will be quite thick). Don’t worry if your design looks random and spotted. Bake brownies until center is set and no longer looks wet, 22–25 minutes. Transfer pan to a wire rack and let cool. Using parchment paper overhang, lift brownies out of pan and transfer to a cutting board. Remove parchment paper and cut into sixteen 2¼\" squares, wiping knife clean between slices.\n\nThis is a other recipe with 10 ingredients.",
    "Recipe: Killer Chocolate Cake\nCategory: cake\nComplexity: 16 ingredients, 23 steps\n\nIngredients needed:\n1 cup (2 sticks) plus 1 tablespoon unsalted butter, at room temperature, 2 1/3 cups all-purpose flour, 1/2 cup Dutch-process cocoa powder, 1 1/2 teaspoons baking powder, 1/2 teaspoon baking soda, 1/2 teaspoon fine sea salt, 1 3/4 cups buttermilk, 2 teaspoons pure vanilla extract, 2 1/4 cups light brown sugar, 3 large eggs, 8 ounces semisweet or bittersweet chocolate, finely chopped, and melted (see Note), 10 tablespoons (1 1/4 sticks) unsalted butter, at room temperature, 1 cup confectioners’ sugar, 4 ounces bittersweet chocolate, finely chopped, melted, and cooled to room temperature (see Note), 2 tablespoons soy sauce, 1 teaspoon pure vanilla extract\n\nCooking instructions:\nMake the cake: Preheat the oven to 350°F. Grease a 9-by-13-inch glass or metal cake pan with the 1 tablespoon of room temperature butter. Whisk together the flour, cocoa, baking powder, baking soda, and salt in a large bowl. Combine the buttermilk and vanilla in a medium bowl or a 2-cup liquid measuring cup. In the bowl of a stand mixer fitted with the paddle attachment (or in a large bowl, if using a handheld mixer), cream the 1 cup of butter and the brown sugar on low speed until creamy and well combined. Increase the speed to medium-high and beat until light and airy, about 2 minutes. Reduce the mixer speed to medium-low and add the eggs one at a time, mixing well between each addition and scraping down the side and bottom of the bowl as needed. Once all 3 eggs are added, beat for 1 minute on medium speed to get the mixture nice and fluffy. Reduce the speed to medium-low and add the flour mixture alternately with the buttermilk mixture in three batches, starting with the flour. Add the melted chocolate and mix on medium speed until well incorporated, stopping the mixer to scrape down the side and bottom of the bowl as needed. Use a rubber or offset spatula to scrape the batter into the prepared pan and even it out as much as possible. Bake until a cake tester inserted into the center of the cake comes out clean and the center of the cake resists light pressure, about 40 minutes. Remove from the oven and set aside to cool completely, at least 2 hours, before frosting. Make the frosting: Put the butter in the bowl of a stand mixer fitted with the whisk attachment (or in a medium bowl, if using a handheld mixer). Beat on medium-high speed until smooth. Turn off the mixer and sift the confectioners’ sugar into the bowl and combine on low speed. Add the melted chocolate, soy sauce, and vanilla and beat on low speed until combined. Increase the mixer speed to medium-low and whip until glossy, 15 to 30 seconds. Use an offset spatula or butter knife to spread the frosting on top of the cooled cake. Cut into squares and serve. DO AHEAD: The cake will keep, loosely covered with plastic wrap in the refrigerator, for up to 3 days. Let it sit out at room temperature for 15 to 20 minutes before slicing and serving.\n\nThis is a cake recipe with 16 ingredients.",
    "Recipe: Chocolate-Almond Butter Cups\nCategory: other\nComplexity: 8 ingredients, 13 steps\n\nIngredients needed:\n1/2 cup smooth almond butter, 1 tablespoon pure maple syrup, 1/2 teaspoon ground cinnamon, 1/4 teaspoon sea salt, 1 tablespoon coconut flour, 2 cups bittersweet chocolate chips (or chopped bittersweet chocolate), 1 1/2 tablespoons coconut oil, Flaky sea salt, such as Maldon, for garnish\n\nCooking instructions:\nLine a mini muffin pan with paper liners and set aside. In a bowl, whisk together the almond butter, maple syrup, cinnamon, sea salt, and coconut flour until well combined; a smooth, thickish paste should form. Cover and chill in the refrigerator or freezer until thickened; 10 to 15 minutes. Remove from the freezer and roll into a log. Break into 24 equal pieces and mold each piece into a flat disk slightly smaller than the cups in the muffin pan. Set aside. Melt together the chocolate chips and coconut oil over a double boiler or in a microwave in 10-second increments, stirring until smooth. Drop a teaspoon of melted chocolate into each of the prepared cups, and shake/tap the pan to create a smooth, even layer. Place the disks of filling on top of the chocolate, and top with the remaining chocolate. Tap to eliminate air bubbles. Sprinkle with the flaky salt and freeze until set. Let come to room temperature before serving.\n\nThis is a other recipe with 8 ingredients."
  ],
  "response": "I'm excited to help you with your chocolate cake baking! Here are two delicious chocolate cake recipes you can try:\n\n1. Recipe Name: Chocolatey Chocolate Cake\nIngredients:\n- 1 3/4 cups (225g) all-purpose flour\n- 1/2 cup (45g) unsweetened Dutch-processed cocoa powder\n- 1 1/2 teaspoons baking powder\n- 1 1/2 teaspoons baking soda\n- 1/2 teaspoon salt\n- 1 1/2 cups (300g) granulated sugar\n- 2 eggs, at room temperature\n- 1 cup (240ml) whole milk\n- 1/2 cup (120ml) grapeseed oil or any mild-flavored oil\n- 1/2 teaspoon pure vanilla extract\n- 1 cup (240ml) boiling water\n\nInstructions:\n- Preheat the oven to 350°F. Grease and line a round pan.\n- Sift dry ingredients, whisk wet ingredients, combine, and add boiling water.\n- Bake for about 50 minutes and let cool before serving.\n\n2. Recipe Name: Killer Chocolate Cake\nIngredients:\n- 1 cup (2 sticks) plus 1 tablespoon unsalted butter\n- 2 1/3 cups all-purpose flour\n- 1/2 cup Dutch-process cocoa powder\n- 1 1/2 teaspoons baking powder\n- 1/2 teaspoon baking soda\n- 1 3/4 cups buttermilk\n- 2 teaspoons pure vanilla extract\n- 2 1/4 cups light brown sugar\n- 3 large eggs\n- 8 ounces semisweet or bittersweet chocolate\n\nInstructions:\n- Cream butter and sugar, add eggs, mix dry and wet ingredients, incorporate melted chocolate.\n- Bake at 350°F for about 40 minutes, then frost and serve.\n\nI hope you enjoy baking these chocolate cakes! Let me know if you need any more help or tips. Happy baking!"
}
```

Response en un formato mas legible:

```markdown
I'm excited to help you with your chocolate cake baking! Here are two delicious 
chocolate cake recipes you can try:

1. Recipe Name: Chocolatey Chocolate Cake
Ingredients:
- 1 3/4 cups (225g) all-purpose flour
- 1/2 cup (45g) unsweetened Dutch-processed cocoa powder
- 1 1/2 teaspoons baking powder
- 1 1/2 teaspoons baking soda
- 1/2 teaspoon salt
- 1 1/2 cups (300g) granulated sugar
- 2 eggs, at room temperature
- 1 cup (240ml) whole milk
- 1/2 cup (120ml) grapeseed oil or any mild-flavored oil
- 1/2 teaspoon pure vanilla extract
- 1 cup (240ml) boiling water

Instructions:
- Preheat the oven to 350°F. Grease and line a round pan.
- Sift dry ingredients, whisk wet ingredients, combine, and add boiling water.
- Bake for about 50 minutes and let cool before serving.

2. Recipe Name: Killer Chocolate Cake
Ingredients:
- 1 cup (2 sticks) plus 1 tablespoon unsalted butter
- 2 1/3 cups all-purpose flour
- 1/2 cup Dutch-process cocoa powder
- 1 1/2 teaspoons baking powder
- 1/2 teaspoon baking soda
- 1 3/4 cups buttermilk
- 2 teaspoons pure vanilla extract
- 2 1/4 cups light brown sugar
- 3 large eggs
- 8 ounces semisweet or bittersweet chocolate

Instructions:
- Cream butter and sugar, add eggs, mix dry and wet ingredients, incorporate melted chocolate.
- Bake at 350°F for about 40 minutes, then frost and serve.

I hope you enjoy baking these chocolate cakes! Let me know if you need any 
more help or tips. Happy baking!"

```

## Ejemplo 2: Pregunta vacía:

Query: (contiene un espacio)

JSON devuelto:

```json
{
  "query": " ",
  "documents": [],
  "response": "Please provide a valid query."
}
```

## Módulo orchestrator.py

Acá también decidí definir una clase para el agente orquestador: `AgentOrchestrator`. Para implementarlo me basé en la siguiente documentación de LlamaIndex: [https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#pattern-2--orchestrator-agent-sub-agents-as-tools](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#pattern-2--orchestrator-agent-sub-agents-as-tools).

La implementación **asume** que ya existe el índice y la base de datos vectorial, por lo tanto no chequea su existencia y reutiliza las creadas por el rag.py.

Explico los constructores de la clase acá:

### \_\_init\_\_

En esta función:

- Se carga el modelo LLM y el modelo de embeddings de OpenAI.
- Se conecta a la base vectorial persistente (ChromaDB) ya creada.
- Se carga un índice ya generado (`load_index_from_storage`).
- Se inicializa un `FunctionAgent`, que actuará como **agente orquestador**.

```python
def __init__(self, # hyperparameters, to make it easier to change if needed
							chroma_path: str = "./chromadb",
							index_path: str = "./index_storage",
							collection_name: str = "dessert_recipes",
							llm_model: str = "gpt-3.5-turbo",
							embedding_model: str = "text-embedding-3-small",
							top_k: int = 5,
            ):

	Settings.llm = OpenAI(model=llm_model, api_key=openai_api_key)
	Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
	self.top_k = top_k

	# --- the following assumes the index and chroma db already exist! --- #
	# initializes ChromaDB
	self.chroma_client = chromadb.PersistentClient(path=chroma_path)
	self.collection = self.chroma_client.get_collection(collection_name)
	
	# loads the vector store with the ChromaDB collection
	self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
	
	# loads the index from the persisted storage
	self.storage_context = StorageContext.from_defaults(
		persist_dir=index_path,
		vector_store=self.vector_store,
	)
	
	self.index = load_index_from_storage(self.storage_context)

	# builds orchestrator agent
	self.agent = self._build_orchestrator()
```

Al último se llama a _build_orchestrator:

### _build_orchestrator

Otro constructor de la clase:

- define las herramientas del agente orquestador
- el prompt del agente

Utilizo `Context` de LlamaIndex para manejar los estados de las herramientas y `FunctionAgent` para construirlo.

Las herramientas son:

- `is_english` → devuelve un booleano. Chequea si la query esta en inglés o no. Cambia dos estados:
    - el estado `"is_english"` setea en `true` o `false` según corresponda
    - el estado `"original_query"` guarda la query original.
    
    Devuelve el booleano.
    
- `translate_query` → traduce o no (según corresponda) la query. Cambia el estado `"translated_query"`.
    - Si el estado `"is_english"` es `true`, no traduce y guarda en `"translated_query"` la query original.
    - Caso contrario, traduce y utiliza el llm para hacerlo. Guarda en `"translated_query"` la query en inglés.
    
    Devuelve la query traducida o no.
    
- `get_documents` → indexa documentos usando _as_query_engine básico de LlamaIndex.
    - Usa la query traducida para indexar.
    - Guarda en el estado `"documents"` los documentos indexados.
    
    Devuelve la lista de documentos.
    
- `generate_response` → obtiene del estado los documentos, la query en inglés y genera una respuesta llamando al llm.
    - Guarda la respuesta en el estado `"answer"`.

```python
def _build_orchestrator(self) -> FunctionAgent:
	"""Builds the orchestrator agent.
	I used the FunctionAgent class from LlamaIndex.
	It has the following tools:
		- is_english: checks if the query is in English
		- translate_query: translates the query to English
		- get_documents: queries the index and retrieves the documents
		- generate_response: generates the response for the query
	I used the Context class from LlamaIndex to pass the state to the tools.
	"""
	# TOOLS =====================================================
	
	async def is_english(ctx: Context, query: str) -> bool:
		"""Simple heuristic to check if text is likely English. This saves a call to the OpenAI api"""
		try:
			is_english = detect(query) == 'en'
			
			# save information in the state
			state = await ctx.store.get("state")
			state["is_english"] = is_english
			state["original_query"] = query
			await ctx.store.set("state", state)
			
			return is_english
		except Exception as e:
			return None

    async def translate_query(ctx: Context) -> str:
        """Translates the query given by the user to English because that's the language
        the recipes were embedded in. This maintains consistency and allows for better
        retrieval of the recipes.
        
        translated_query has the original query if the query is already in English,
        otherwise it has the translated query."""

        state = await ctx.store.get("state")
        original_query = state.get("original_query")

        # if the query is already in English, returns it
        if state.get("is_english"):
            english_query = original_query
        else:
            # if the query is not in English, translates it
            translation_prompt = f"""
                Translate the following query into English as accurately as possible.
                Return ONLY the translated text, no explanations, no additional commentary.
                Preserve the original meaning exactly, without adding, removing, or altering 
                any information.

                Query: {original_query}

                English translation:
            """ # same prompt as the one used in the RAG system
            
            translation = Settings.llm.complete(translation_prompt)
            english_query = str(translation).strip()
            
        # save information in the state
        state["translated_query"] = english_query
        await ctx.store.set("state", state)

        return english_query
    
    async def get_documents(ctx: Context) -> List[str]:
        """Retrieves documents from the vector store based on the query"""

        state = await ctx.store.get("state")
        english_query = state.get("translated_query")

        # query the index and retrieve the documents
        results = self.index.as_query_engine(similarity_top_k=self.top_k).query(english_query)
        source_nodes = results.source_nodes
        documents = [node.node.get_content() for node in source_nodes]

        # save information in the state
        state["documents"] = documents
        await ctx.store.set("state", state)

        return documents

    async def generate_response(ctx: Context) -> str:
        """Generates the final response using the retrieved documents"""

        state = await ctx.store.get("state")
        documents = state.get("documents")

        if not documents:
            response_text = "I couldn't find any relevant recipes for your query."
            state["answer"] = response_text
            await ctx.store.set("state", state)
            return response_text
        
        english_query = state.get("translated_query")

        prompt = f"""
            Answer the following question: {english_query} using the documents below.

            Here are the documents that may contain the answer:
            {documents}

            Please provide a helpful response based on the documents above. 
            If the documents don't contain relevant information, say so.
        """

        # generate the response
        response = Settings.llm.complete(prompt=prompt)
        response_text = str(response).strip()

        # save information in the state
        state["answer"] = response_text
        await ctx.store.set("state", state)

        return response_text

        tools = [
            is_english,
            translate_query,
            get_documents,
            generate_response,
        ]

        system_prompt = (
            "You are an orchestrator agent for a dessert recipe assistant. "
            "You must coordinate tools to detect language, translate queries if needed, "
            "retrieve recipes from a vector store, and generate a final response. "
            "Follow this workflow: 1) Check if query is English, 2) Translate if needed, "
            "3) Get relevant documents, 4) Generate response based on documents. "
            "Only use available tools. Once you've generated the response, return it."
        )

        return FunctionAgent(
            tools=tools,
            llm=Settings.llm,
            system_prompt=system_prompt,
            initial_state={
                "is_english": None,
                "original_query": None,
                "translated_query": None,
                "query_results": None,
                "documents": [],
                "answer": None
            }
        )
```

Por último tenemos el constructor `run` que corre el agente inicializado y devuelve su respuesta.

```python
async def run(self, query: str) -> str:
	"""Run the orchestrator agent asynchronously."""
	# basic validation
	query_strip = query.strip()
	if not query_strip:
		return "Please provide a valid query."
	
	try:
		response = await self.agent.run(query_strip)
		# returns the agent's response directly
		return str(response)
		
	except Exception as e:
		return f"Sorry, an error occurred: {str(e)}"
```

## Ejemplo 1: Pregunta sobre tortas de chocolate

Query: What chocolate cakes can I bake?

JSON devuelto:

```json
{
  "response": "Based on the provided documents, you can bake the following chocolate cakes:\n1. Chocolatey Chocolate Cake\n2. Chocolate Zucchini Cake\n3. Killer Chocolate Cake\n4. Full Moon Chocolate Zucchini Cake\n5. Crunchy Chocolate Caramel Layer Cake\n\nThese are the chocolate cake recipes mentioned in the documents provided. Each recipe has its own set of ingredients and cooking instructions for you to try out. Enjoy baking!"
}
```

Response en un formato mas legible:

```markdown
Based on the provided documents, you can bake the following 
chocolate cakes:

1. Chocolatey Chocolate Cake  
2. Chocolate Zucchini Cake  
3. Killer Chocolate Cake  
4. Full Moon Chocolate Zucchini Cake  
5. Crunchy Chocolate Caramel Layer Cake

These are the chocolate cake recipes mentioned in the documents provided. 
Each recipe has its own set of ingredients and cooking instructions for you to 
try out. Enjoy baking!
```

## Ejemplo 2: Pregunta vacía:

Query: (contiene un espacio)

JSON devuelto:

```markdown
{
  "response": "Please provide a valid query."
}
```

# Algunas consideraciones
* Intenté buscar la base de datos vectorial de Groq pero no encontre, lo que si encontré es que Groq tiene un modelo de embedding. Notar que no es dificil cambiar la implementación para usar ese modelo, ya que es simplemente cambiar el hiperparámetro de embedding_model al de Groq, y el cliente de OpenAIEmbedding al cliente que tenga Groq para hacerlo.
