export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function percentile(sorted, fraction) {
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function summarize(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.length > 1
    ? samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
      (samples.length - 1)
    : 0;
  return {
    median_ms: Number(median(samples).toFixed(6)),
    mean_ms: Number(mean.toFixed(6)),
    stddev_ms: Number(Math.sqrt(variance).toFixed(6)),
    p95_ms: Number(percentile(sorted, 0.95).toFixed(6)),
    min_ms: Number(sorted[0].toFixed(6)),
    max_ms: Number(sorted[sorted.length - 1].toFixed(6)),
    samples_ms: samples.map((value) => Number(value.toFixed(6))),
  };
}
