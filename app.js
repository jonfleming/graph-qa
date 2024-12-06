import { OpenAIEmbeddings } from "@langchain/openai";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import 'dotenv/config'

// Configuration object for Neo4j connection and other related settings
const config = {
  url: process.env.NEO4J_URI, // URL for the Neo4j instance
  username: process.env.NEO4J_USERNAME, // Username for Neo4j authentication
  password: process.env.NEO4J_PASSWORD, // Password for Neo4j authentication
  indexName: "vector", // Name of the vector index
  keywordIndexName: "keyword", // Name of the keyword index if using hybrid search
  searchType: "vector", // Type of search (e.g., vector, hybrid)
  nodeLabel: "Chunk", // Label for the nodes in the graph
  textNodeProperty: "text", // Property of the node containing text
  embeddingNodeProperty: "embedding", // Property of the node containing embedding
};

const documents = [
  { pageContent: "what's this", metadata: { a: 2 } },
  { pageContent: "Cat drinks milk", metadata: { a: 1 } },
];

const neo4jVectorIndex = await Neo4jVectorStore.fromDocuments(
  documents,
  new OpenAIEmbeddings(),
  config
);

const results = await neo4jVectorIndex.similaritySearch("a thing", 1);

console.log(results);

/*
  [ Document { pageContent: 'Cat drinks milk', metadata: { a: 1 } } ]
*/

await neo4jVectorIndex.close();