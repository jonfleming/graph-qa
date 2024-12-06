import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { GraphCypherQAChain } from "@langchain/community/chains/graph_qa/cypher";
//import { GraphCypherQAChain } from "langchain/chains/graph_qa/cypher";
import 'dotenv/config'

const NEO4J_URL = process.env.NEO4J_URI
const NEO4J_USERNAME = process.env.NEO4J_USERNAME
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD

// Configuration object for Neo4j connection and other related settings
const config = {
  url: NEO4J_URL, // URL for the Neo4j instance
  username: NEO4J_USERNAME, // Username for Neo4j authentication
  password: NEO4J_PASSWORD, // Password for Neo4j authentication
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

let neo4jVectorIndex, neo4jGraph

async function init() {
  neo4jVectorIndex = await Neo4jVectorStore.fromDocuments(
    documents,
    new OpenAIEmbeddings(),
    config
  );

  neo4jGraph = await Neo4jGraph.initialize({ url: NEO4J_URL, username: NEO4J_USERNAME, password: NEO4J_PASSWORD});
}

async function vectorSearch() {
  const results = await neo4jVectorIndex.similaritySearch("water", 1);
  console.log(results);
/*
  [ Document { pageContent: 'Cat drinks milk', metadata: { a: 1 } } ]
*/
  await neo4jVectorIndex.close();
}

async function cypherQuery() {
  // QUERY = """
  //   "MATCH (m:Movie)-[:IN_GENRE]->(:Genre {name:$genre})
  //   RETURN m.title, m.plot
  //   ORDER BY m.imdbRating DESC LIMIT 5"
  //   """
  //
  // await neo4jGraph.query(QUERY, genre="action")

  const QUERY = `MATCH (n) RETURN n LIMIT 1`;
  const results = await neo4jGraph.query(QUERY);
  console.log(results);
}

async function graphChain() {
  // Populate the database with two nodes and a relationship
  await neo4jGraph.query(`
    CREATE (a:Actor {name:'Bruce Willis'})
    -[:ACTED_IN]->(:Movie {title: 'Pulp Fiction'})
  `);

  await neo4jGraph.refreshSchema();

  const model = new OpenAI({ temperature: 0 });
  const chain = GraphCypherQAChain.fromLLM({ llm: model, graph: neo4jGraph });
  const res = await chain.run("Who acted in Pulp Fiction?");

  console.log(res);
}

async function main() {
  await init();
  await vectorSearch();
  await cypherQuery();
  await graphChain();
}

main();


