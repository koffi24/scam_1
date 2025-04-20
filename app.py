from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from anti_scam import context, arnaque, instructions, negative, lapsus, llm_client

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('send_message')
def handle_message(data):
    user_role = data['role']  # 'user1' (arnaqueur) ou 'user2' (victime)
    user_message = data['message']

    # Diffuser le message de l'utilisateur à tous les clients
    emit('receive_message', {'role': user_role, 'message': user_message}, broadcast=True)

    if user_role == 'user2':  # Si c'est la victime qui envoie un message
        # Ajouter le contexte et les instructions au message
        messages = [
            {"role": "system", "content": context + "\n" + arnaque + "\n" + instructions + "\n" + negative + "\n" + lapsus},
            {"role": "user", "content": user_message}
        ]
        completion = llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct",
            messages=messages,
            max_tokens=1512,
            stream=False,
            temperature=0.9
        )
        response = completion.choices[0].message.content

        # Envoyer la réponse de l'arnaqueur (user1) au client
        emit('receive_message', {'role': 'user1', 'message': response}, broadcast=True)

#if __name__ == '__main__':
   #socketio.run(app, debug=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)