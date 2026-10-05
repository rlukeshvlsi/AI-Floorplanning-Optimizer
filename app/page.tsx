"use client";

import { useState } from "react";

type Placement = {
  x: number;
  y: number;
};

type Result = {
  success: boolean;
  error?: string;
  filename?: string;

  initial_placement?: Record<string, Placement>;
  optimized_placement?: Record<string, Placement>;

  initial_qor?: number;
  qor_before_refinement?: number;
  qor_after_refinement?: number;
  improvement?: number;
  refinement_improvement?: number;
  predicted_best_qor?: number;

  candidates_generated?: number;
  ml_screening_evaluations?: number;
  refinement_evaluations?: number;
  total_evaluations?: number;
  evaluation_reduction?: number;

  optimized_image?: string;
};

/*
 * IMPORTANT:
 * This is intentionally /api/optimize.
 *
 * Local:
 *     FastAPI service handles /api/optimize
 *
 * Vercel:
 *     Vercel Services routes /api/* to the FastAPI service.
 */
const API_URL = "/api/optimize";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const selected = event.target.files?.[0];

    if (!selected) {
      return;
    }

    if (!selected.type.startsWith("image/")) {
      setError("Please upload a PNG or JPG floorplan image.");
      return;
    }

    setFile(selected);
    setResult(null);
    setError("");

    const url = URL.createObjectURL(selected);
    setPreview(url);
  }

  async function optimizeFloorplan() {
    if (!file) {
      setError("Please upload a floorplan image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data: Result = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Optimization failed."
        );
      }

      setResult(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Could not connect to the AI optimizer.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">

      {/* HEADER */}

      <header className="border-b border-slate-800 bg-slate-950">

        <div className="mx-auto max-w-7xl px-6 py-6">

          <h1 className="text-3xl font-bold tracking-tight">
            AI Floorplanning Optimizer
          </h1>

          <p className="mt-2 text-slate-400">
            ML-assisted VLSI macro floorplan optimization
          </p>

        </div>

      </header>


      {/* MAIN */}

      <section className="mx-auto max-w-7xl px-6 py-10">

        {/* UPLOAD SECTION */}

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-8 shadow-xl">

          <h2 className="text-2xl font-semibold">
            Upload Floorplan
          </h2>

          <p className="mt-2 text-slate-400">
            Upload your existing floorplan image. The system
            detects the macro positions and uses the trained
            Random Forest model to optimize the placement.
          </p>


          {/* FILE UPLOAD */}

          <label className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950 p-10 transition hover:border-blue-500">

            <div className="text-center">

              <div className="mb-3 text-5xl">
                📤
              </div>

              <div className="text-xl font-semibold">
                Choose Floorplan Image
              </div>

              <div className="mt-2 text-sm text-slate-500">
                PNG or JPG
              </div>

            </div>

            <input
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
              className="hidden"
            />

          </label>


          {/* FILE NAME */}

          {file && (

            <div className="mt-4 rounded-lg bg-slate-800 px-4 py-3 text-sm text-slate-300">

              Selected file:

              <span className="ml-2 font-semibold text-white">
                {file.name}
              </span>

            </div>

          )}


          {/* PREVIEW */}

          {preview && (

            <div className="mt-8">

              <h3 className="mb-3 text-lg font-semibold">
                Uploaded Floorplan
              </h3>

              <div className="overflow-hidden rounded-xl border border-slate-700 bg-white">

                <img
                  src={preview}
                  alt="Uploaded floorplan"
                  className="mx-auto max-h-[550px] w-auto object-contain"
                />

              </div>

            </div>

          )}


          {/* BUTTON */}

          <button
            onClick={optimizeFloorplan}
            disabled={!file || loading}
            className="mt-6 w-full rounded-xl bg-blue-600 px-6 py-4 text-lg font-semibold transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-slate-700"
          >

            {loading
              ? "Running AI Optimization..."
              : "🚀 Optimize Floorplan"}

          </button>


          {/* ERROR */}

          {error && (

            <div className="mt-5 rounded-xl border border-red-800 bg-red-950/40 p-4 text-red-300">

              <strong>Error:</strong> {error}

            </div>

          )}

        </div>


        {/* RESULTS */}

        {result?.success && (

          <div className="mt-10 space-y-8">

            {/* RESULT HEADER */}

            <div>

              <h2 className="text-3xl font-bold">
                Optimization Results
              </h2>

              <p className="mt-2 text-slate-400">
                Random Forest candidate screening followed by
                actual QoR evaluation and local refinement.
              </p>

            </div>


            {/* METRICS */}

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">

              <Metric
                title="Initial QoR"
                value={result.initial_qor}
              />

              <Metric
                title="Optimized QoR"
                value={result.qor_after_refinement}
              />

              <Metric
                title="QoR Improvement"
                value={result.improvement}
              />

              <Metric
                title="Evaluation Reduction"
                value={result.evaluation_reduction}
                suffix="%"
              />

            </div>


            {/* IMAGE COMPARISON */}

            <div className="grid gap-6 lg:grid-cols-2">

              {/* ORIGINAL */}

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

                <h3 className="mb-4 text-xl font-semibold">
                  Original Floorplan
                </h3>

                <div className="overflow-hidden rounded-xl bg-white">

                  {preview && (

                    <img
                      src={preview}
                      alt="Original floorplan"
                      className="mx-auto max-h-[550px] w-auto object-contain"
                    />

                  )}

                </div>

              </div>


              {/* OPTIMIZED */}

              <div className="rounded-2xl border border-blue-800 bg-slate-900 p-6">

                <h3 className="mb-4 text-xl font-semibold text-blue-400">
                  Optimized Floorplan
                </h3>

                <div className="overflow-hidden rounded-xl bg-white">

                  {result.optimized_image && (

                    <img
                      src={result.optimized_image}
                      alt="Optimized floorplan"
                      className="mx-auto max-h-[550px] w-auto object-contain"
                    />

                  )}

                </div>

              </div>

            </div>


            {/* PLACEMENT TABLES */}

            <div className="grid gap-6 lg:grid-cols-2">

              <PlacementCard
                title="Initial Placement"
                placement={result.initial_placement}
              />

              <PlacementCard
                title="Optimized Placement"
                placement={result.optimized_placement}
              />

            </div>


            {/* OPTIMIZATION DETAILS */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

              <h3 className="mb-6 text-xl font-semibold">
                Optimization Details
              </h3>

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

                <Detail
                  label="Candidates Generated"
                  value={result.candidates_generated}
                />

                <Detail
                  label="ML Screening Evaluations"
                  value={result.ml_screening_evaluations}
                />

                <Detail
                  label="Refinement Evaluations"
                  value={result.refinement_evaluations}
                />

                <Detail
                  label="Total Evaluations"
                  value={result.total_evaluations}
                />

                <Detail
                  label="Predicted Best QoR"
                  value={result.predicted_best_qor}
                />

                <Detail
                  label="Refinement Improvement"
                  value={result.refinement_improvement}
                />

              </div>

            </div>


            {/* PIPELINE */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

              <h3 className="mb-6 text-xl font-semibold">
                AI Optimization Pipeline
              </h3>

              <div className="grid gap-4 md:grid-cols-5">

                <Step
                  number="1"
                  title="Upload"
                  text="Floorplan image"
                />

                <Step
                  number="2"
                  title="Detect"
                  text="Macro coordinates"
                />

                <Step
                  number="3"
                  title="Predict"
                  text="Random Forest"
                />

                <Step
                  number="4"
                  title="Evaluate"
                  text="Top candidates"
                />

                <Step
                  number="5"
                  title="Refine"
                  text="Local optimization"
                />

              </div>

            </div>

          </div>

        )}

      </section>

    </main>
  );
}


/* ============================================================
   METRIC
   ============================================================ */

function Metric({
  title,
  value,
  suffix = "",
}: {
  title: string;
  value?: number;
  suffix?: string;
}) {

  return (

    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

      <p className="text-sm text-slate-500">
        {title}
      </p>

      <p className="mt-2 text-3xl font-bold">
        {value !== undefined
          ? `${value}${suffix}`
          : "--"}
      </p>

    </div>

  );
}


/* ============================================================
   DETAIL
   ============================================================ */

function Detail({
  label,
  value,
}: {
  label: string;
  value?: number;
}) {

  return (

    <div className="rounded-xl bg-slate-950 p-4">

      <p className="text-sm text-slate-500">
        {label}
      </p>

      <p className="mt-1 text-xl font-semibold">
        {value !== undefined
          ? value
          : "--"}
      </p>

    </div>

  );

}


/* ============================================================
   PLACEMENT CARD
   ============================================================ */

function PlacementCard({
  title,
  placement,
}: {
  title: string;
  placement?: Record<string, Placement>;
}) {

  return (

    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

      <h3 className="mb-5 text-xl font-semibold">
        {title}
      </h3>

      <div className="overflow-hidden rounded-xl border border-slate-800">

        <table className="w-full">

          <thead className="bg-slate-800">

            <tr>

              <th className="px-4 py-3 text-left">
                Macro
              </th>

              <th className="px-4 py-3 text-left">
                X
              </th>

              <th className="px-4 py-3 text-left">
                Y
              </th>

            </tr>

          </thead>

          <tbody>

            {["A", "B", "C", "D"].map((name) => (

              <tr
                key={name}
                className="border-t border-slate-800"
              >

                <td className="px-4 py-3 font-semibold">
                  {name}
                </td>

                <td className="px-4 py-3">
                  {placement?.[name]?.x ?? "--"}
                </td>

                <td className="px-4 py-3">
                  {placement?.[name]?.y ?? "--"}
                </td>

              </tr>

            ))}

          </tbody>

        </table>

      </div>

    </div>

  );

}


/* ============================================================
   PIPELINE STEP
   ============================================================ */

function Step({
  number,
  title,
  text,
}: {
  number: string;
  title: string;
  text: string;
}) {

  return (

    <div className="rounded-xl bg-slate-950 p-5">

      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 font-bold">
        {number}
      </div>

      <h4 className="font-semibold">
        {title}
      </h4>

      <p className="mt-1 text-sm text-slate-500">
        {text}
      </p>

    </div>

  );

}