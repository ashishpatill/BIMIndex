"use client";

import { useMemo, useState } from "react";
import { Filter, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type GraphSection = { name?: string; chunk_ids?: string[]; document_ids?: string[]; pages?: number[] };
type GraphEntity = { name?: string; chunk_ids?: string[]; document_ids?: string[]; pages?: number[] };
type GraphReference = { reference?: string; source_chunk_ids?: string[]; target_chunk_ids?: string[]; document_ids?: string[] };
type KnowledgeGraph = {
  stats?: {
    node_count?: number; edge_count?: number; section_count?: number;
    entity_count?: number; reference_count?: number; relation_counts?: Record<string, number>;
  };
  sections?: GraphSection[]; entities?: GraphEntity[]; references?: GraphReference[];
};
type Props = { graph: KnowledgeGraph | null };
type ArtifactKind = "sections" | "entities" | "references";

export function KnowledgeGraphInspector({ graph }: Props) {
  const [artifactKind, setArtifactKind] = useState<ArtifactKind>("entities");
  const [relationFilter, setRelationFilter] = useState("all");
  const [query, setQuery] = useState("");

  const relationRows = useMemo(
    () => Object.entries(graph?.stats?.relation_counts ?? {}).sort((a, b) => b[1] - a[1]),
    [graph],
  );
  const artifactRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const items = graph?.[artifactKind] ?? [];
    if (!normalized) return items;
    return items.filter((row) => JSON.stringify(row).toLowerCase().includes(normalized));
  }, [artifactKind, graph, query]);

  if (!graph?.stats) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Knowledge graph</CardTitle>
          <div className="flex items-center gap-2">
            <Select value={relationFilter} onValueChange={setRelationFilter}>
              <SelectTrigger className="h-8 w-36 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All relations</SelectItem>
                {relationRows.map(([rel]) => (
                  <SelectItem key={rel} value={rel}>{rel}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="relative w-40">
              <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Filter ${artifactKind}`}
                className="h-8 pl-7 text-xs"
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-5">
          <StatCard label="Nodes" value={graph.stats.node_count ?? 0} />
          <StatCard label="Edges" value={graph.stats.edge_count ?? 0} />
          <StatCard label="Sections" value={graph.stats.section_count ?? 0} />
          <StatCard label="Entities" value={graph.stats.entity_count ?? 0} />
          <StatCard label="References" value={graph.stats.reference_count ?? 0} />
        </div>

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

        <Tabs value={artifactKind} onValueChange={(v) => setArtifactKind(v as ArtifactKind)}>
          <TabsList>
            <TabsTrigger value="entities">Entities</TabsTrigger>
            <TabsTrigger value="sections">Sections</TabsTrigger>
            <TabsTrigger value="references">References</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="grid gap-2 md:grid-cols-2">
          {artifactRows.slice(0, 24).map((row, idx) => (
            <ArtifactCard key={`${artifactKind}-${idx}`} kind={artifactKind} row={row} />
          ))}
          {!artifactRows.length && (
            <p className="col-span-full py-8 text-center text-sm text-muted-foreground">No matching artifacts.</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}

function ArtifactCard({ kind, row }: { kind: ArtifactKind; row: GraphSection | GraphEntity | GraphReference }) {
  const title =
    kind === "references" ? (row as GraphReference).reference : (row as GraphSection | GraphEntity).name;
  const chunkCount =
    kind === "references"
      ? ((row as GraphReference).source_chunk_ids?.length ?? 0) +
        ((row as GraphReference).target_chunk_ids?.length ?? 0)
      : ((row as GraphSection | GraphEntity).chunk_ids?.length ?? 0);
  const pages = "pages" in row ? row.pages ?? [] : [];

  return (
    <div className="rounded-lg border bg-card p-3">
      <p className="truncate text-sm font-medium">{title ?? "(untitled)"}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        {chunkCount} chunks{pages.length ? ` | pages ${pages.slice(0, 6).join(", ")}` : ""}
      </p>
      {(row.document_ids?.length ?? 0) > 0 && (
        <p className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
          {row.document_ids?.join(", ")}
        </p>
      )}
    </div>
  );
}
