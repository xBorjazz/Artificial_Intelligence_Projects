% Cargar el archivo FIS
fis = readfis('Practica1.fis');

% Definir entradas de prueba
input1 = 25; % Por ejemplo, una entrada de error
input2 = -10; % Por ejemplo, una entrada de rate

% Evaluar el sistema difuso
output = evalfis([input1, input2], fis);

% Mostrar el resultado
disp(['El resultado es: ', num2str(output)]);
