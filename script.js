document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const userInput = document.getElementById('user_input');
    const userForm = document.getElementById('userForm');
    
    let isRecording = false;

    recordButton.addEventListener('click', function() {
        if (!isRecording) {
            startRecording();
        }
    });

    stopButton.addEventListener('click', function() {
        stopRecording();
    });

    function startRecording() {
        isRecording = true;
        recordButton.classList.add('recording');
        recordButton.disabled = true; // Disable to prevent repeated clicks

        fetch('/start_listening', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            isRecording = false;
            recordButton.classList.remove('recording');
            recordButton.disabled = false;

            if (data.text && data.text !== "Could not understand audio") {
                userInput.value = data.text;
            } else {
                alert("Speech not recognized. Please try again.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            isRecording = false;
            recordButton.classList.remove('recording');
            recordButton.disabled = false;
        });
    }

    function stopRecording() {
        if (isRecording) {
            isRecording = false;
            recordButton.classList.remove('recording');
            recordButton.disabled = false;
        }
    }

    userForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const customerName = document.getElementById('customer_name').value;
        const userInputText = userInput.value;

        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                customer_name: customerName,
                user_input: userInputText
            })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('sentiment').textContent = 
                `Sentiment: ${data.sentiment.sentiment} (Polarity: ${data.sentiment.polarity})`;
            document.getElementById('response').textContent = 
                `AI Response: ${data.llama_response}`;
        })
        .catch(error => console.error('Error:', error));
    });
});
