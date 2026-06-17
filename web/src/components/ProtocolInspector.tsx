import React, { useState, useRef, useEffect } from 'react';
import type { ProtocolEvent, StructureField } from '../protocol/inspect';
import {
  tagName,
  parseStructure,
  formatBytes,
} from '../protocol/inspect';

// Color palette for structure field highlighting
const FIELD_COLORS = [
  '#93c5fd', // blue-300
  '#86efac', // green-300
  '#c4b5fd', // violet-300
  '#fcd34d', // amber-300
  '#fca5a5', // red-300
  '#67e8f9', // cyan-300
  '#f9a8d4', // pink-300
  '#bef264', // lime-300
  '#fdba74', // orange-300
  '#d8b4fe', // purple-300
];

function fieldColor(index: number): string {
  return FIELD_COLORS[index % FIELD_COLORS.length];
}

/** Build a byte-offset → color map from structure fields. */
function createByteColorMap(fields: StructureField[]): Map<number, string> {
  const map = new Map<number, string>();
  fields.forEach((field, i) => {
    const color = fieldColor(i);
    const end = Math.min(field.offset + field.length, field.offset + 2048);
    for (let j = field.offset; j < end; j++) {
      map.set(j, color);
    }
  });
  return map;
}

// --- Sub-components ---

function EventRow({
  event,
  selected,
  onClick,
  baseTime,
}: {
  event: ProtocolEvent;
  selected: boolean;
  onClick: () => void;
  baseTime: number;
}) {
  const relTime = ((event.timestamp - baseTime) / 1000).toFixed(3);
  const isSent = event.direction === 'sent';

  return (
    <div
      className={`px-3 py-1.5 cursor-pointer text-xs font-mono border-b border-gray-800 transition-colors
        ${selected ? 'bg-gray-700' : 'hover:bg-gray-800/60'}`}
      onClick={onClick}
    >
      <div className="flex items-center gap-1.5">
        <span className="text-gray-600 w-16 text-right shrink-0">
          +{relTime}s
        </span>
        <span
          className={`shrink-0 ${isSent ? 'text-blue-400' : 'text-green-400'}`}
        >
          {isSent ? '\u2191' : '\u2193'}
        </span>
        <span className="text-gray-500 w-3 shrink-0">{event.server}</span>
        <span className="text-gray-200 flex-1 truncate">
          {tagName(event.tag)}
        </span>
        <span className="text-gray-600 shrink-0">
          {formatBytes(event.totalBytes)}
        </span>
      </div>
    </div>
  );
}

function HexDump({
  bytes,
  fields,
  totalBytes,
}: {
  bytes: Uint8Array;
  fields: StructureField[];
  totalBytes: number;
}) {
  const colorMap = createByteColorMap(fields);
  const defaultColor = '#6b7280';
  const rows: React.ReactNode[] = [];

  for (let offset = 0; offset < bytes.length; offset += 16) {
    const hexCells: React.ReactNode[] = [];
    const asciiCells: React.ReactNode[] = [];

    for (let i = 0; i < 16; i++) {
      if (i === 8) {
        hexCells.push(
          <span key="gap" className="inline-block w-2" />
        );
      }
      const byteIdx = offset + i;
      if (byteIdx >= bytes.length) {
        hexCells.push(
          <span key={`h${i}`} className="invisible">
            {'00 '}
          </span>
        );
        asciiCells.push(<span key={`a${i}`}>{' '}</span>);
      } else {
        const b = bytes[byteIdx];
        const color = colorMap.get(byteIdx) || defaultColor;
        hexCells.push(
          <span key={`h${i}`} style={{ color }}>
            {b.toString(16).padStart(2, '0')}{' '}
          </span>
        );
        const ch = b >= 32 && b < 127 ? String.fromCharCode(b) : '.';
        asciiCells.push(
          <span key={`a${i}`} style={{ color }}>
            {ch}
          </span>
        );
      }
    }

    rows.push(
      <div key={offset} className="flex whitespace-pre leading-5">
        <span className="text-gray-600 w-20 select-none shrink-0">
          {offset.toString(16).padStart(8, '0')}
          {'  '}
        </span>
        <span className="shrink-0">{hexCells}</span>
        <span className="text-gray-700 select-none shrink-0">{'  |'}</span>
        <span className="shrink-0">{asciiCells}</span>
        <span className="text-gray-700 select-none">{'|'}</span>
      </div>
    );
  }

  return (
    <div className="font-mono text-xs">
      <div className="flex whitespace-pre leading-5 text-gray-600 border-b border-gray-800 pb-1 mb-1">
        <span className="w-20 shrink-0">{'Offset  '}</span>
        <span className="shrink-0">
          {'00 01 02 03 04 05 06 07'}
          <span className="inline-block w-2" />
          {'08 09 0a 0b 0c 0d 0e 0f'}
        </span>
        <span className="shrink-0">{'   ASCII'}</span>
      </div>
      {rows}
      {totalBytes > bytes.length && (
        <div className="text-gray-500 mt-2 text-xs">
          {'\u2026 '}{formatBytes(totalBytes - bytes.length)} more ({formatBytes(totalBytes)} total)
        </div>
      )}
    </div>
  );
}

function StructurePanel({
  fields,
  event,
}: {
  fields: StructureField[];
  event: ProtocolEvent;
}) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-300 mb-1">
        {tagName(event.tag)}
      </h3>
      <p className="text-[11px] text-gray-500 mb-3">
        {event.direction === 'sent' ? 'Sent to' : 'Received from'} Server{' '}
        {event.server} &middot; {formatBytes(event.totalBytes)}
      </p>
      <div className="space-y-2">
        {fields.map((field, i) => (
          <div
            key={i}
            className={`rounded-lg p-2.5 text-xs border ${
              field.opaque
                ? 'bg-red-950/30 border-red-900/40'
                : 'bg-gray-800/60 border-gray-700/40'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: fieldColor(i) }}
              />
              <span className="font-medium text-gray-200">{field.name}</span>
              {field.opaque && (
                <span className="ml-auto px-1.5 py-0.5 bg-red-900/50 text-red-400 rounded text-[10px] font-semibold tracking-wide uppercase shrink-0">
                  Opaque
                </span>
              )}
            </div>
            <div className="text-gray-500 mb-1 font-mono">
              [{field.offset}..{field.offset + field.length}) &middot;{' '}
              {field.length} byte{field.length !== 1 ? 's' : ''}
            </div>
            <div className="text-gray-300 font-mono break-all leading-relaxed">
              {field.value}
            </div>
            {field.description && (
              <div className="text-gray-500 mt-1.5 text-[11px] leading-snug">
                {field.description}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Main Inspector ---

interface Props {
  events: ProtocolEvent[];
}

export default function ProtocolInspector({ events }: Props) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Auto-scroll event list to bottom when events arrive
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [events.length]);

  // Auto-select first event if none selected
  useEffect(() => {
    if (selectedId === null && events.length > 0) {
      setSelectedId(events[0].id);
    }
  }, [events.length, selectedId]);

  const selectedEvent = events.find((e) => e.id === selectedId) ?? null;
  const fields = selectedEvent ? parseStructure(selectedEvent) : [];
  const baseTime = events[0]?.timestamp ?? 0;

  return (
    <div className="flex flex-1 min-h-0">
      {/* Event list */}
      <div
        ref={listRef}
        className="w-64 shrink-0 border-r border-gray-700 overflow-y-auto"
      >
        {events.length === 0 ? (
          <div className="p-4 text-gray-500 text-xs leading-relaxed">
            No protocol events yet.
            <br />
            Send a message to observe the 2PC data flow.
          </div>
        ) : (
          events.map((event) => (
            <EventRow
              key={event.id}
              event={event}
              selected={selectedId === event.id}
              onClick={() => setSelectedId(event.id)}
              baseTime={baseTime}
            />
          ))
        )}
      </div>

      {/* Hex dump */}
      <div className="flex-1 overflow-auto p-4 min-w-0">
        {selectedEvent ? (
          <HexDump
            bytes={selectedEvent.rawBytes}
            fields={fields}
            totalBytes={selectedEvent.totalBytes}
          />
        ) : (
          <div className="text-gray-500 text-sm">
            Select an event to view raw data
          </div>
        )}
      </div>

      {/* Structure sidebar */}
      <div className="w-80 shrink-0 border-l border-gray-700 overflow-y-auto p-4">
        {selectedEvent ? (
          <StructurePanel fields={fields} event={selectedEvent} />
        ) : (
          <div className="text-gray-500 text-sm">
            Select an event to see its structure
          </div>
        )}
      </div>
    </div>
  );
}
