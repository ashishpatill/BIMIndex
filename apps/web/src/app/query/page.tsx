import { QueryWorkbench } from "@/components/query-workbench";

export default function QueryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Query</h1>
        <p className="mt-1 text-muted-foreground">Search documents and inspect retrieval results.</p>
      </div>
      <QueryWorkbench />
    </div>
  );
}
