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

type DataPoint = { timestamp: number; [key: string]: any };

interface TimeSeriesChartProps<D> {
  newDataPoint: D;
  getData: (data: D) => number;
  axisLabel: string;
  paused?: boolean;
  objectId?: string | number;
  maxDataPoints?: number;
  className?: string;
}

function TimeSeriesChart<D extends DataPoint>({
  newDataPoint,
  getData,
  axisLabel,
  paused = false,
  objectId = 0,
  maxDataPoints = 100,
  className = "",
}: TimeSeriesChartProps<D>) {
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
    if (!paused) {
      setData((oldData) =>
        [...oldData, newDataPoint].slice(-maxDataPointsRef.current),
      );
    }
  }, [newDataPoint, paused]);

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
              value: axisLabel,
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
            labelFormatter={(label, _) =>
              `Time: ${formatXAxis(label as number)}`
            }
            formatter={(value) =>
              `${
                typeof value === "number" ? value.toFixed(2) : value
              } ${axisLabel}`
            }
          />
          <Line
            type="monotone"
            dataKey={(d) => (d !== null ? getData(d) : null)}
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
