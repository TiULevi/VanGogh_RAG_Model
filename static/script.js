document.addEventListener('DOMContentLoaded', function() {
    const sendButton = document.getElementById('sendButton');
    sendButton.addEventListener('click', askQuestion);
});

async function askQuestion() {
    const question = document.getElementById("question").value;
    const messagesDiv = document.getElementById("messages");

    try {
        // Send the question to the /ask route with POST request
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question, language: 'English' })
        });

        if (!response.ok) throw new Error('Failed to fetch answer from server');

        // Parse the result from the server response
        const result = await response.json();
        const botResponse = result.answer;

        // Display the user's question and the bot's response
        messagesDiv.innerHTML += `<div class="message user">${question}</div>`;
        messagesDiv.innerHTML += `<div class="message bot">${botResponse}</div>`;

        // Clear the input field
        document.getElementById("question").value = '';
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        messagesDiv.innerHTML += `<div class="message bot">Sorry, something went wrong.</div>`;
    }
}


