export class GraphNLPClient {
  constructor(private apiKey: string, private baseUrl = "https://api.graphnlp.io") {}

  async analyze(documents: string[], domain = "generic") {
    const r = await fetch(`${this.baseUrl}/v1/analyze`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ documents, domain }),
    });
    return r.json();
  }

  async getGraph(graphId: string) {
    const r = await fetch(`${this.baseUrl}/v1/graph/${graphId}`,
      { headers: { "Authorization": `Bearer ${this.apiKey}` } });
    return r.json();
  }
}
