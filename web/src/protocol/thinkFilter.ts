/**
 * Strip <think>...</think> blocks from text.
 *
 * Handles both complete blocks and incomplete blocks at the end
 * (during streaming, the closing tag may not have arrived yet).
 */
export function stripThinkBlocks(text: string): string {
  // Strip complete think blocks
  let result = text.replace(/<think>[\s\S]*?<\/think>\s*/g, '');
  // Strip incomplete think block at the end (still streaming)
  result = result.replace(/<think>[\s\S]*$/, '');
  return result;
}
