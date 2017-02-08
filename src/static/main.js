
    var selectedHeroes = []
    var heroesLeft = 10

    function decreaseCounter(){
        var e = document.getElementById("counter")
        var num = e.getElementsByTagName("p")[0]
        heroesLeft -= 1
        num.innerHTML = heroesLeft
    }

    function myFunction(elem){
        decreaseCounter()
        var heroID = elem.alt
        selectedHeroes.push(heroID)
        document.getElementById(heroID).className += " selected"
        console.log(selectedHeroes)

    }

    function post(path, params, method) {
        method = method || "post"; // Set method to post by default if not specified.

        // The rest of this code assumes you are not using a library.
        // It can be made less wordy if you use one.
        var form = document.createElement("form");
        form.setAttribute("method", method);
        form.setAttribute("action", path);

        for(var key in params) {
            if(params.hasOwnProperty(key)) {
                var hiddenField = document.createElement("input");
                hiddenField.setAttribute("type", "hidden");
                hiddenField.setAttribute("name", key);
                hiddenField.setAttribute("value", params[key]);

                form.appendChild(hiddenField);
             }
        }

        document.body.appendChild(form);
        form.submit();
    }

    function sendPost() {
        var xmlhttp = new XMLHttpRequest();   // new HttpRequest instance
        xmlhttp.open("POST", "/results");
        xmlhttp.setRequestHeader("Content-Type", "application/json");
        xmlhttp.send(JSON.stringify(selectedHeroes));
    }


    function predict() {
        if (selectedHeroes.length < 5) {
            alert("Not enough heroes selected!")
        }
        else if(selectedHeroes.length > 10) {
            alert("Too many heroes selected. Clearing selections...")
            selectedHeroes = []
            location.reload()
        }
        else { // make prediction
            sendPost()
            window.location.href = "/results";
        }
     }
