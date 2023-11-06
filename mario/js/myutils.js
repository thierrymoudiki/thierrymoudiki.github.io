var myElement = document.getElementById('update-note');

function colorChange() {
    if (myElement.innerText == "Loading, please wait...") {  
        myElement.style.color = 'orange';
    } else {
        myElement.style.color = 'green';
    }
  }