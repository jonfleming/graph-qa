import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { ChatOllama } from '@langchain/ollama'
import { Neo4jVectorStore } from '@langchain/community/vectorstores/neo4j_vector'
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph'
import { GraphCypherQAChain } from '@langchain/community/chains/graph_qa/cypher'
import { LLMGraphTransformer } from '@langchain/community/experimental/graph_transformers/llm'
import { OllamaEmbeddings } from '@langchain/ollama'
import { Document } from '@langchain/core/documents'

import 'dotenv/config'

const NEO4J_URL = process.env.NEO4J_URI
const NEO4J_USERNAME = process.env.NEO4J_USERNAME
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD
const EMBEDDING_MODEL = 'gpt-4o' // 'nomic-embed-text'
const LLM_MODEL = 'text-embedding-ada-002' // 'llama3.2'
const LLM_BASE_URL = 'http://localhost:11434'
const llm = new OpenAI({ temperature: 0 }) // const model = new ChatOllama({ model: LLM_MODEL, temperature: 0 })

// Configuration object for Neo4j connection and other related settings
const config = {
  url: NEO4J_URL, // URL for the Neo4j instance
  username: NEO4J_USERNAME, // Username for Neo4j authentication
  password: NEO4J_PASSWORD, // Password for Neo4j authentication
  indexName: 'vector', // Name of the vector index
  keywordIndexName: 'keyword', // Name of the keyword index if using hybrid search
  searchType: 'vector', // Type of search (e.g., vector, hybrid)
  nodeLabel: 'Chunk', // Label for the nodes in the graph
  textNodeProperty: 'text', // Property of the node containing text
  embeddingNodeProperty: 'embedding' // Property of the node containing embedding
}

const embeddings = new OpenAIEmbeddings() // const embeddings = new OllamaEmbeddings({ model: EMBEDDING_MODEL, baseUrl: LLM_BASE_URL})

const documents = [
  { pageContent: "what's this", metadata: { a: 2 } },
  { pageContent: 'Cat drinks milk', metadata: { a: 1 } }
]

let neo4jVectorIndex, neo4jGraph

async function init () {
  neo4jGraph = await Neo4jGraph.initialize({
    url: NEO4J_URL,
    username: NEO4J_USERNAME,
    password: NEO4J_PASSWORD
  })
  await dropAll()

  neo4jVectorIndex = await Neo4jVectorStore.fromDocuments(
    documents,
    embeddings,
    config
  )
}

async function dropIndex () {
  // Drop index if it exists
  console.log('Dropping index if it exists')
  const QUERY = `DROP INDEX ${config.indexName} IF EXISTS`
  const results = await neo4jGraph.query(QUERY)
  console.log(results)
}

async function dropAll () {
  // Drop index if it exists
  await dropIndex()

  // Drop all nodes and relationships in the database
  console.log('Dropping all nodes and relationships in the database')
  const result = await neo4jGraph.query('MATCH (n) DETACH DELETE n')
  console.log(result)
}

async function vectorSearch () {
  const results = await neo4jVectorIndex.similaritySearch('water', 1)
  console.log(results)
  /*
  [ Document { pageContent: 'Cat drinks milk', metadata: { a: 1 } } ]
*/
  await neo4jVectorIndex.close()
}

async function cypherQuery () {
  // QUERY = """
  //   "MATCH (m:Movie)-[:IN_GENRE]->(:Genre {name:$genre})
  //   RETURN m.title, m.plot
  //   ORDER BY m.imdbRating DESC LIMIT 5"
  //   """
  //
  // await neo4jGraph.query(QUERY, genre="action")

  const QUERY = `MATCH (n) RETURN n LIMIT 1`
  const results = await neo4jGraph.query(QUERY)
  console.log(results)
}

async function graphChain () {
  // Populate the database with two nodes and a relationship
  await neo4jGraph.query(`
    CREATE (a:Actor {name:'Bruce Willis'})
    -[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})
  `)

  await neo4jGraph.refreshSchema()

  const chain = GraphCypherQAChain.fromLLM({ llm: llm, graph: neo4jGraph })
  const res = await chain.run('Who acted in Pulp Fiction?')

  console.log(res)
}

async function insertGraphNodes () {
  // Insert knowledge into the graph
  // https://js.langchain.com/docs/how_to/graph_constructing
  const transformer = new ChatOpenAI({ temperature: 0, model: 'gpt-4o-mini' })
  const llmGraphTransformer = new LLMGraphTransformer({ llm: transformer })

  let text = `
  Marie Curie, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
  She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
  Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
  She was, in 1906, the first woman to become a professor at the University of Paris.
  `

  const graphDocuments = await llmGraphTransformer.convertToGraphDocuments([
    new Document({ pageContent: text })
  ])

  await neo4jGraph.addGraphDocuments(graphDocuments, {baseEntityLabel: true, includeSource: true});
  await neo4jGraph.close()
}

async function main () {
  await init()
  // await vectorSearch()
  // await cypherQuery()
  // await graphChain()
  await insertGraphNodes()
}

main()
