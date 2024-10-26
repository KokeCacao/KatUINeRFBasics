import { useMemo } from 'react';
import { NodeProps } from 'reactflow';
import GsplatScene from '../../frontend/GsplatScene';
import { NodeDefaultBody } from '../../frontend/components/NodeDefaultBody';
import { NodeDefaultCard } from '../../frontend/components/NodeDefaultCard';
import { NodeDefaultOutput } from '../../frontend/components/NodeDefaultOutput';

export function RenderGaussian(node: NodeProps) {
  const viewPort = useMemo(() => {
    if (node.data.output.gaussian === undefined) {
      return <GsplatScene />;
    }
    const base64String = node.data.output.gaussian.value;
    const byteCharacters = atob(base64String);
    // Convert the characters to a byte array
    const byteArray = new Uint8Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteArray[i] = byteCharacters.charCodeAt(i);
    }
    const blob = new Blob([byteArray], { type: 'application/octet-stream' });
    const file = new File([blob], 'gaussian.ply', {
      type: 'application/octet-stream',
    });
    return <GsplatScene file={file} />;
  }, [node.data.output.gaussian]);

  const body = useMemo(
    () => (
      <NodeDefaultBody
        nodeId={node.id}
        nodeType={node.type}
        nodeDataInput={node.data.input}
        nodeDataInputEdgeIds={node.data.input_edge_ids}
      />
    ),
    [node.id, node.type, node.data.input, node.data.input_edge_ids],
  );

  const items = useMemo(
    () => [
      {
        name: 'Input',
        children: (
          <div className="node-body">
            {body}
            {viewPort}
          </div>
        ),
      },
      {
        name: 'Output',
        children: (
          <>
            <div className="node-body-hidden">{body}</div>
            <div className="node-output">
              <NodeDefaultOutput nodeDataOutput={node.data.output} />
            </div>
          </>
        ),
      },
      {
        name: 'Collapse',
        children: <div className="node-body-hidden">{body}</div>,
      },
    ],
    [body, node.data.output, viewPort],
  );

  const defaultCard = useMemo(
    () => (
      <>
        <NodeDefaultCard
          nodeId={node.id}
          nodeType={node.type}
          nodeDataInput={node.data.input}
          nodeDataOutput={node.data.output}
          nodeDataState={node.data.state}
          nodeDataCachePolicy={node.data.cache_policy}
          items={items}
        />
      </>
    ),
    [
      node.id,
      node.type,
      node.data.input,
      node.data.output,
      node.data.state,
      node.data.cache_policy,
      items,
    ],
  );

  return defaultCard;
}
