import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface TimeSeriesChartProps<D, K extends keyof D> {
  newDataPoint: D;
  selectedKey: K;
  objectId?: string | number;
  transform?: (value: D[K]) => number;
  axisLabels?: { [K in keyof D]?: string };
  maxDataPoints?: number;
  className?: string;
}

function TimeSeriesChart<D extends { timestamp: number }, K extends keyof D>({
  newDataPoint,
  selectedKey,
  axisLabels,
  objectId = 0,
  transform = (v) => Number(v),
  maxDataPoints = 100,
  className = "",
}: TimeSeriesChartProps<D, K>) {
  const [currentObjectId, setObjectId] = useState<string | number>(objectId);
  const [firstTs, setFirstTs] = useState<number>(newDataPoint.timestamp);
  const [data, setData] = useState<D[]>([]);

  const maxDataPointsRef = useRef(maxDataPoints);
  maxDataPointsRef.current = maxDataPoints;

  useEffect(() => {
    if (objectId !== currentObjectId) {
      setObjectId(objectId);
      setFirstTs(newDataPoint.timestamp);
      setData([]);
    }
  }, [objectId, currentObjectId]);

  useEffect(() => {
    setData((oldData) =>
      [...oldData, newDataPoint].slice(-maxDataPointsRef.current)
    );
  }, [newDataPoint]);

  const formatXAxis = (timestamp: number) => {
    return (timestamp - firstTs).toFixed(1);
  };

  return (
    <div className={cn("w-full h-64 bg-white rounded-lg px-2", className)}>
      <ResponsiveContainer
        width="100%"
        height="100%"
        className="ml-[-15px] pt-4"
      >
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tick={false} />
          <YAxis
            label={{
              value: axisLabels?.[selectedKey] || selectedKey,
              angle: -90,
              position: "insideLeft",
              offset: 15,
            }}
            padding={{
              bottom: 0,
              top: 0,
            }}
          />
          <Tooltip
            labelStyle={{ color: "black" }}
            labelFormatter={(label, val) =>
              `Time: ${formatXAxis(label as number)}`
            }
            formatter={(value) =>
              `${typeof value === "number" ? value.toFixed(2) : value}${
                " " + axisLabels?.[selectedKey] ?? ""
              }`
            }
          />
          <Line
            type="monotone"
            dataKey={(data) => transform(data[selectedKey as string])}
            stroke="black"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default TimeSeriesChart;
