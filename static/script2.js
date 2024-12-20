document.addEventListener('DOMContentLoaded', function() {
    const sendButton = document.getElementById('sendButton');
    sendButton.addEventListener('click', askQuestion);
});

async function askQuestion() {
    const question = document.getElementById("question").value;
    const messagesDiv = document.getElementById("messages");

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!response.ok) throw new Error('Failed to fetch answer from server');

        const result = await response.json();
        const botResponse = result.answer;

        messagesDiv.innerHTML += `<div class="message user">${question}</div>`;
        messagesDiv.innerHTML += `<div class="message bot">${botResponse}</div>`;

        document.getElementById("question").value = '';
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        messagesDiv.innerHTML += `<div class="message bot">Sorry, something went wrong.</div>`;
    }
}
