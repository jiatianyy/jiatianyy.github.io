export default function responsiveTables() {
  return function visit(node) {
    if (!node.children) return;
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      if (child.type === 'element' && child.tagName === 'table') {
        node.children[i] = { type: 'element', tagName: 'div', properties: { className: ['table-scroll'], tabIndex: 0, role: 'region', ariaLabel: 'Scrollable comparison table' }, children: [child] };
      } else visit(child);
    }
  };
}
