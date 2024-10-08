% Routines for opening audio files and convert them to feature vectors
% by means of periodogram.
%
% Last modification: 20/01/2022
% Author: Guilherme Barreto

clear; clc; close all

%pkg load image
pkg load financial

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fase 1 -- Carrega arquivo de audio disponiveis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
part0 = 'kit/'
part1 = 'comando_';
part2 = {'avancar_0' 'direita_0' 'esquerda_0' 'parar_0' 'recuar_0'};
part3 = strvcat(part2);
part4 = '.wav';

Ncom=5;   % Quantidade de comandos
Nreal=10;  % Quantidade de realizacoes

X=[];   % Matriz de dados (vetores de atributos)
Y=[];   % Matriz de rotulos (one-hot encoding);

for i=1:Ncom,  % Indice para os comandos
    Comando=i,
    for j=1:Nreal,   % Indice para expressoes
        nome = strcat(part0,part1,part3(i,:),int2str(j),part4);    % Monta o nome do arquivo de audio

        [sinal Fs]=audioread(nome);

        % Etapa 2: SUBAMOSTRAGEM
        N=length(sinal);
        sinal1=sinal(:,1);   % sinal original canal 1
        sinal2=sinal(:,2);   % sinal original canal 2

        Ipar=2:2:N;
        Iimpar=1:2:N-1;

        sinal1a = sinal1(Iimpar);  % 1o. sinal subamostrado do canal 1
        sinal1b = sinal1(Ipar);    % 2o. sinal subamostrado do canal 1
        sinal2a = sinal2(Iimpar);  % 1o. sinal subamostrado do canal 2
        sinal2b = sinal2(Ipar);    % 2o. sinal subamostrado do canal 2

        % Etapa 3: PERIODOGRAMA
        [P1a W]=periodogram(sinal1a,[],[],Fs/2);
        [P1b W]=periodogram(sinal1b,[],[],Fs/2);
        [P2a W]=periodogram(sinal2a,[],[],Fs/2);
        [P2b W]=periodogram(sinal2b,[],[],Fs/2);
        %figure; plot(W,P1a);

        % Etapa 4: GERAR VETORES DE ATRIBUTOS
        Li=[0 100 200 360 500 700 850 1000 1200 1320 1500 2200 2700];
        Ls=[Li(2:end) 3700];

        % Monta vetor de atributos a partir dos periodogramas
        Nint=length(Li);  % No. de faixas de frequencias escolhidas
        for l=1:Nint,
          I=find(W>=Li(l) & W<=Ls(l));   % l-esima banda de frequencia
          x1a(l)=max(P1a(I));    % Pega maior potencia dentro da l-esima
          x1b(l)=max(P1b(I));    % Pega maior potencia dentro da l-esima
          x2a(l)=max(P2a(I));    % Pega maior potencia dentro da l-esima
          x2b(l)=max(P2b(I));    % Pega maior potencia dentro da l-esima
        endfor

        X=[X x1a(:) x1b(:) x2a(:) x2b(:)];
        %ROT=zeros(Ncom,1); ROT(i)=1;  % Cria rotulo binario do vetor de atributos (one-hot encoding)
        ROT = i;
        Y=[Y ROT ROT ROT ROT];
    end
end

%%%%%%%% APLICACAO DE BOX-COX %%%%%%%%%%%

lambda = 0.5;  % Example value, can be changed for optimization

% Apply the Box-Cox transformation manually
for idx = 1:size(X, 1)
    if lambda == 0
        X(idx, :) = log(X(idx, :) + eps);  % Log transformation for lambda = 0
    else
        X(idx, :) = (X(idx, :) .^ lambda - 1) / lambda;  % Box-Cox transformation
    end
end

%%%%%%%% APLICACAO DE PCA %%%%%%%%%%%
[V L VEi]=pcacov(cov(X'));
q=4; Vq=V(:,1:q); Qq=Vq'; X=Qq*X;
VEq=cumsum(VEi); figure; plot(VEq,'r-','linewidth',3);
xlabel('Autovalor');
ylabel('Variancia explicada acumulada');


Z=[X;Y];  % Formato 01 vetor de atributos por coluna: DIM(Z) = (p+1)xN
Z=Z';     % Formato 01 vetor de atributos por linha: DIM(Z) = Nx(p+1)

save -ascii recvoz.dat Z

##save -ascii comandos_input.txt X
##save -ascii comandos_output.txt Y
