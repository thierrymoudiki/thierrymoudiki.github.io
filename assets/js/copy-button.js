document.addEventListener('DOMContentLoaded', function () {
  const codeBlocks = document.querySelectorAll('.post-content pre > code');

  codeBlocks.forEach(function (codeBlock) {
    const pre = codeBlock.parentNode;
    pre.style.position = 'relative';

    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.innerText = 'Copy';

    // Insert the button at the top right of the <pre>
    pre.appendChild(button);

    button.addEventListener('click', function () {
      navigator.clipboard.writeText(codeBlock.innerText).then(() => {
        button.innerText = 'Copied!';
        setTimeout(() => {
          button.innerText = 'Copy';
        }, 2000);
      });
    });
  });
});