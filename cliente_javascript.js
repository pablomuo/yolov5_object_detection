const WebSocket = require("ws");
const socket = new WebSocket('ws://10.236.28.136:8000');


// Abre la conexión
socket.addEventListener('open', function (event) {
    console.log("Connected with server. YOLO:");
});


// Escucha por mensajes
socket.addEventListener('message', function (event) {
    console.log(event.data);
});


// para ejecutar este código escribir en el términal:
// node cliente_javascript.js
