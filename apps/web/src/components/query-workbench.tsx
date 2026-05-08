"use client";

import { useState } from "react";
import {
  BarChart3,
  Eye,
  FlaskConical,
  Network,
  Search,
  Send,
  Settings2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/input";
import { apiBaseUrl } from "@/lib/api";

type ResultPayload = {
  run_id: string;
  result: {
    answer: string;
    knowledge_card?: {
      confidence?: number;
      answerable?: boolean;
      answerability_reason?: string;
      unresolved_ambiguity?: string[];
      follow_up_retrieval_suggestions?: string[];
    } & Record<string, unknown>;
    evidence?: EvidenceItem[];
  };
  trace: { steps?: TraceStep[] } & Record<string, unknown>;
};

type EvidenceItem = {
  chunk_id?: string;
  retrieval_path?: string;
  score?: number;
  metadata?: {
    graph_relations?: string[];
    graph_expanded_from?: string[];
    image_path?: string;
    visual_profile?: string[];
  } & Record<string, unknown>;
};

type TraceStep = {
  path?: string;
  document_id?: string;
  document_ids?: string[];
  hits?: number;
  diagnostics?: {
    node_count?: number;
    edge_count?: number;
    seed_count?: number;
    expanded_count?: number;
    relation_counts?: Record<string, number>;
    expanded_relation_counts?: Record<string, number>;
    query_entities?: string[];
    query_references?: string[];
  };
  [key: string]: unknown;
};

function GraphDiagnostics({ payload }: { payload: ResultPayload }) {
  const [relationFilter, setRelationFilter] = useState("all");
  const graphSteps = (payload.trace.steps ?? []).filter(
    (step) => (step.path === "graph" || step.path === "graph_corpus") && step.diagnostics,
  );
  const graphEvidence = (payload.result.evidence ?? []).filter(
    (item) => item.metadata?.graph_relations?.length,
  );

  if (!graphSteps.length && !graphEvidence.length) return null;

  const relationCounts = graphSteps.reduce<Record<string, number>>((acc, step) => {
    const counts =
      step.diagnostics?.expanded_relation_counts ?? step.diagnostics?.relation_counts ?? {};
    for (const [rel, count] of Object.entries(counts)) {
      acc[rel] = (acc[rel] ?? 0) + count;
    }
    return acc;
  }, {});
  const relationRows = Object.entries(relationCounts).sort((a, b) => b[1] - a[1]);
  const filteredEvidence =
    relationFilter === "all"
      ? graphEvidence
      : graphEvidence.filter((item) => item.metadata?.graph_relations?.includes(relationFilter));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="h-4 w-4" />
            Graph diagnostics
          </CardTitle>
          {relationRows.length > 0 && (
            <Select value={relationFilter} onValueChange={setRelationFilter}>
              <SelectTrigger className="h-8 w-40 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All relations</SelectItem>
                {relationRows.map(([rel]) => (
                  <SelectItem key={rel} value={rel}>{rel}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {graphSteps.map((step, idx) => (
          <div
            key={`${step.document_id ?? step.document_ids?.join("-") ?? "doc"}-${idx}`}
            className="grid gap-2 sm:grid-cols-4"
          >
            <div className="rounded-lg border bg-card p-2">
              <p className="text-[10px] text-muted-foreground">Nodes</p>
              <p className="text-sm font-semibold">{step.diagnostics?.node_count ?? 0}</p>
            </div>
            <div className="rounded-lg border bg-card p-2">
              <p className="text-[10px] text-muted-foreground">Edges</p>
              <p className="text-sm font-semibold">{step.diagnostics?.edge_count ?? 0}</p>
            </div>
            <div className="rounded-lg border bg-card p-2">
              <p className="text-[10px] text-muted-foreground">Seeds</p>
              <p className="text-sm font-semibold">{step.diagnostics?.seed_count ?? 0}</p>
            </div>
            <div className="rounded-lg border bg-card p-2">
              <p className="text-[10px] text-muted-foreground">Expanded</p>
              <p className="text-sm font-semibold">{step.diagnostics?.expanded_count ?? 0}</p>
            </div>
          </div>
        ))}
        {relationRows.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {relationRows
              .filter(([rel]) => relationFilter === "all" || rel === relationFilter)
              .map(([rel, count]) => (
                <Badge key={rel} variant="secondary" className="text-[10px]">
                  {rel}: {count}
                </Badge>
              ))}
          </div>
        )}
        {filteredEvidence.length > 0 && (
          <div className="space-y-1.5">
            {filteredEvidence.slice(0, 5).map((item) => (
              <div key={item.chunk_id} className="rounded-lg border bg-card p-2 text-xs">
                <p className="font-medium">{item.chunk_id}</p>
                <p className="mt-0.5 text-muted-foreground">
                  Relations: {item.metadata?.graph_relations?.join(", ")}
                </p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function VisualDiagnostics({ payload }: { payload: ResultPayload }) {
  const visualEvidence = (payload.result.evidence ?? []).filter((item) =>
    item.retrieval_path?.includes("visual"),
  );
  const visualSteps = (payload.trace.steps ?? []).filter((step) => step.path?.startsWith("visual"));
  if (!visualEvidence.length && !visualSteps.length) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-4 w-4" />
          Visual diagnostics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Visual steps: {visualSteps.length} | Evidence: {visualEvidence.length}
        </p>
        <div className="space-y-1.5">
          {visualEvidence.slice(0, 5).map((item) => (
            <div key={item.chunk_id} className="rounded-lg border bg-card p-2 text-xs">
              <p className="font-medium">
                {item.chunk_id} ({item.retrieval_path ?? "visual"})
              </p>
              <p className="text-muted-foreground">Score: {(item.score ?? 0).toFixed(3)}</p>
              {item.metadata?.visual_profile?.length && (
                <p className="text-muted-foreground">Profile: {item.metadata.visual_profile.join(", ")}</p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function QueryWorkbench() {
  const [question, setQuestion] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [mode, setMode] = useState("planner");
  const [topK, setTopK] = useState(5);
  const [plannerMergeStrategy, setPlannerMergeStrategy] = useState("score_max");
  const [plannerRerank, setPlannerRerank] = useState(true);
  const [plannerRouteVoteBonus, setPlannerRouteVoteBonus] = useState(0.08);
  const [plannerRerankOverlapWeight, setPlannerRerankOverlapWeight] = useState(0.1);
  const [showPlannerOptions, setShowPlannerOptions] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState<ResultPayload | null>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setError("");
    setPayload(null);
    try {
      const response = await fetch(`${apiBaseUrl()}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          document_id: documentId || null,
          mode,
          top_k: topK,
          planner_merge_strategy: plannerMergeStrategy,
          planner_rerank: plannerRerank,
          planner_route_vote_bonus: plannerRouteVoteBonus,
          planner_rerank_overlap_weight: plannerRerankOverlapWeight,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data?.detail || "Query failed.");
      } else {
        setPayload(data as ResultPayload);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Query workbench
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="question">Question</Label>
              <Textarea
                id="question"
                className="min-h-[100px]"
                placeholder="Ask a question about your documents..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                required
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="docId">Document ID (optional)</Label>
                <Input
                  id="docId"
                  placeholder="All documents"
                  value={documentId}
                  onChange={(e) => setDocumentId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="mode">Mode</Label>
                <Select value={mode} onValueChange={setMode}>
                  <SelectTrigger id="mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="planner">planner</SelectItem>
                    <SelectItem value="hybrid">hybrid</SelectItem>
                    <SelectItem value="bm25">bm25</SelectItem>
                    <SelectItem value="dense">dense</SelectItem>
                    <SelectItem value="visual">visual</SelectItem>
                    <SelectItem value="graph">graph</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="topK">Top-K</Label>
                <Input id="topK" type="number" value={topK} min={1} max={20} onChange={(e) => setTopK(Number(e.target.value))} />
              </div>
            </div>
            {mode === "planner" && (
              <>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowPlannerOptions(!showPlannerOptions)}
                  className="flex items-center gap-1 text-xs text-muted-foreground"
                >
                  <Settings2 className="h-3.5 w-3.5" />
                  Planner options
                  {showPlannerOptions ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                </Button>
                {showPlannerOptions && (
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                    <div className="space-y-2">
                      <Label className="text-xs">Merge strategy</Label>
                      <Select value={plannerMergeStrategy} onValueChange={setPlannerMergeStrategy}>
                        <SelectTrigger className="h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="score_max">score_max</SelectItem>
                          <SelectItem value="route_vote">route_vote</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-end gap-2 pb-1">
                      <Checkbox id="rerank" checked={plannerRerank} onCheckedChange={(v) => setPlannerRerank(!!v)} />
                      <Label htmlFor="rerank" className="text-xs font-normal">Query-overlap rerank</Label>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Route vote bonus</Label>
                      <Input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={plannerRouteVoteBonus}
                        onChange={(e) => setPlannerRouteVoteBonus(Number(e.target.value))}
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Rerank overlap weight</Label>
                      <Input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={plannerRerankOverlapWeight}
                        onChange={(e) => setPlannerRerankOverlapWeight(Number(e.target.value))}
                        className="h-8 text-xs"
                      />
                    </div>
                  </div>
                )}
              </>
            )}
            <Button type="submit" disabled={isSubmitting} className="w-full sm:w-auto">
              <Send className="h-4 w-4" />
              {isSubmitting ? "Running..." : "Run query"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="p-4">
            <p className="text-sm text-destructive-foreground">{error}</p>
          </CardContent>
        </Card>
      )}

      {payload && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Answer</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <pre className="whitespace-pre-wrap text-sm">{payload.result.answer}</pre>
              <p className="font-mono text-xs text-muted-foreground">Run ID: {payload.run_id}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Knowledge card
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-lg border bg-card p-3 text-sm">
                <div className="flex items-center gap-2">
                  <Badge variant={payload.result.knowledge_card?.answerable ? "default" : "secondary"}>
                    {String(payload.result.knowledge_card?.answerable ?? false)}
                  </Badge>
                  <span className="text-muted-foreground">
                    Confidence: {(payload.result.knowledge_card?.confidence ?? 0).toFixed(3)}
                  </span>
                </div>
                <p className="mt-2 text-muted-foreground">
                  {payload.result.knowledge_card?.answerability_reason ?? "No reason available."}
                </p>
              </div>
              {(payload.result.knowledge_card?.unresolved_ambiguity?.length ?? 0) > 0 && (
                <div className="rounded-lg border border-amber-900/50 bg-amber-950/30 p-3 text-xs text-amber-200">
                  {payload.result.knowledge_card?.unresolved_ambiguity?.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              )}
              {(payload.result.knowledge_card?.follow_up_retrieval_suggestions?.length ?? 0) > 0 && (
                <div className="rounded-lg border border-sky-900/50 bg-sky-950/30 p-3 text-xs text-sky-200">
                  {payload.result.knowledge_card?.follow_up_retrieval_suggestions?.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <GraphDiagnostics payload={payload} />
          <VisualDiagnostics payload={payload} />

          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Retrieval trace</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap text-xs text-muted-foreground">
                {JSON.stringify(payload.trace, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
