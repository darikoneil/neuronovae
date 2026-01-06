document.addEventListener('DOMContentLoaded', function () {
  const items = document.querySelectorAll('.doc-section-item.field-body');
  items.forEach(li => {
    const nodes = Array.from(li.childNodes);
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (node.nodeType === Node.TEXT_NODE && /default\s*:/i.test(node.textContent)) {
        for (let j = i + 1; j < nodes.length; j++) {
          const next = nodes[j];
          if (next.nodeType === Node.ELEMENT_NODE && next.tagName.toLowerCase() === 'code') {
            next.classList.add('md-default-value');
            break;
          }
        }
        break;
      }
    }
  });

  if (!document.getElementById('md-default-value-style')) {
    const style = document.createElement('style');
    style.id = 'md-default-value-style';
    style.textContent = `
.md-default-value {
  color: var(--md-default-value-color, #c5595e) !important;
  font-weight: 600;
}
`;
    document.head.appendChild(style);
  }
});