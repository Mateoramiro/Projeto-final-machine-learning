PROJETO FINAL: APLICAÇÃO DE MACHINE LEARNING CLÁSSICO
CLASSIFICAÇÃO DE COMPRAS PÚBLICAS COM PREÇO ATÍPICO E PREVISÃO DO PREÇO ESPERADO USANDO MACHINE LEARNING CLÁSSICO
Tema 5.4: Compras.gov.br, compra atípica e preço esperado
Alunos: Mateo Zapelini e Miguel Torres
Professor: Rodrigo Ramos Silva
Disciplina: Machine Learning Clássico
2026

SUMÁRIO
1 INTRODUÇÃO, PROBLEMA E JUSTIFICATIVA	3
1.1 Contextualização do problema real	3
1.2 Definição do problema de Machine Learning	3
1.3 Justificativa e relevância	4
OBJETIVOS	5
1.3.1 Objetivo geral	5
1.3.2 Objetivos específicos	5
1.4 FUNDAMENTAÇÃO TEÓRICA	5
1.4.1 Revisão do domínio: compras públicas e preço de referência	5
1.4.2 Referencial teórico em Machine Learning	7
1.5 METODOLOGIA E DESENVOLVIMENTO	9
1.5.1 Análise Exploratória de Dados (EDA)	9
1.5.2 Pré-processamento e engenharia de atributos	10
1.5.3 Implementação, treinamento e otimização	11
1.5.4 Avaliação e análise de resultados	12
1.5.5 Conclusão, produto final e cronograma	17
REFERÊNCIAS	18

1 INTRODUÇÃO, PROBLEMA E JUSTIFICATIVA
1.1 Contextualização do problema real
Domínio de aplicação. O projeto trata de compras públicas, uma área da administração pública ligada ao controle interno e à auditoria. Todo órgão público, da prefeitura pequena ao ministério, compra o que precisa para funcionar, e a lei exige que o preço pago seja compatível com o que se pratica no mercado (BRASIL, 2021a, art. 23). Escolhemos um item que praticamente todo órgão compra: papel para impressão. Por ser um produto simples e padronizado, fica mais fácil perceber quando um preço sai do padrão.
Situação atual. Hoje o controle do preço é feito de três formas, e todas têm problema:
⦁	Pesquisa de preços manual: antes de comprar, o servidor junta pelo menos três preços e calcula a média, a mediana ou o menor valor, descartando os valores "inexequíveis, inconsistentes e excessivamente elevados" (BRASIL, 2021b, art. 6º). A norma não diz o que é "excessivamente elevado", então a decisão depende de quem faz a pesquisa.
⦁	Consulta a painéis: ferramentas como o Painel de Preços mostram quanto outros órgãos pagaram, mas só respondem a quem pergunta. Não avisam sozinhas que uma compra saiu fora do padrão.
⦁	Auditoria por amostra: os órgãos de controle não conseguem olhar todas as compras. Revisam uma amostra ou as compras mais caras, e o que está fora disso passa sem ser visto.
Há ainda um problema que aparece logo nos dados: o mesmo papel é cadastrado ora por folha, ora por pacote de 50, ora por resma de 500 folhas. Quando a unidade é registrada errado, o preço unitário parece dezenas de vezes maior ou menor do que é. Esses registros entram na base que o próprio governo usa como referência de preço e contaminam as pesquisas seguintes.
Base teórica. O problema é relevante pelo tamanho do mercado envolvido. Ribeiro e Inácio Júnior (2019), em estudo do Ipea, estimaram que as compras governamentais equivalem em média a cerca de 12,5% do PIB brasileiro. E o próprio governo já usa automação para revisar compras: a ferramenta Alice, da Controladoria-Geral da União, analisou mais de 161 mil processos de compra em 2024 e, com ações preventivas, gerou R$ 1,25 bilhão em benefícios financeiros (BRASIL, 2025).
1.2 Definição do problema de Machine Learning
Definição técnica. O projeto tem duas etapas e uma etapa opcional:
1.	Classificação binária: prever se uma compra tem perfil de risco de sair com preço atípico (sim ou não), usando só o que se sabe da compra antes de olhar o preço.
2.	Regressão: estimar o preço esperado da compra, em reais por quilo de papel, a partir das características do item e da compra.
3.	Agrupamento (opcional): usar o K-Means para ver que perfis de compra existem na base.
No fim, as duas primeiras etapas são combinadas numa única medida, o valor em risco: a probabilidade de a compra ser atípica vezes o valor pago acima do esperado. Ordenando as compras por esse valor, temos uma fila de revisão.
Variável alvo. Na classificação, é a coluna compra_atipica, que construímos. Ela vale 1 quando o preço por quilo da compra fica acima de Q3 + 1,5 × IQR do seu tipo de papel, regra sugerida no guia do projeto, e 0 caso contrário. Na regressão, o alvo é o preco_kg, o preço pago por quilo de papel.
Dataset escolhido. Usamos a API de Dados Abertos do Compras.gov.br (COMPRAS.GOV.BR, 2026a), módulo de Pesquisa de Preço, rota 1_consultarMaterial, filtrando o PDM 19746 (papel para impressão formatado) no ano de 2025. Como a coleta foi feita página por página, usamos as 18 primeiras páginas de 75 registros que a API devolve, ou seja, as 1.350 compras mais recentes de 2025, de 8 de agosto a 31 de dezembro. As características de cada papel (tipo, tamanho, gramatura e cor) vieram do catálogo de materiais, rota 4_consultarItemMaterial. O conjunto, que chamamos de compras_papel_2025 (arquivo compras_papel_2025.csv), pode ser obtido pela API em https://dadosabertos.compras.gov.br/swagger-ui/index.html, e os dados abertos de compras do governo federal estão descritos em COMPRAS.GOV.BR (2026b). Acesso em 25 de setembro de 2026.
Escolhemos essa base por quatro motivos. São dados reais e públicos de compras do governo. Cada registro traz preço, quantidade, unidade de fornecimento, órgão comprador e fornecedor, que é tudo o que precisamos para as duas etapas. O papel é comprado por órgãos das três esferas em todos os estados, o que dá variedade. E o próprio guia do projeto indica esta fonte para o tema.
1.3 Justificativa e relevância
Impacto. O modelo transforma milhares de compras soltas numa fila ordenada pelo dinheiro que pode ter sido pago a mais. Com isso, a equipe de controle interno, que não consegue olhar tudo, olha primeiro o que mais importa. Ganha-se em três frentes: a revisão foca onde há dinheiro em jogo, os erros de cadastro são corrigidos antes de contaminar outras pesquisas de preço, e o gestor tem uma referência objetiva do preço esperado.
Relevância do Machine Learning. Por que não resolver isso com uma regra simples, do tipo "revisar tudo que custou mais que o dobro da mediana"? Por três razões.
A primeira é que o preço depende de muitas coisas ao mesmo tempo: tipo de papel, tamanho, gramatura, cor, embalagem, quantidade, região e forma de compra. Uma regra fixa precisaria de uma mediana para cada combinação, e muitas combinações têm poucas compras.
A segunda é que o modelo de regressão aprende o preço esperado mesmo para itens com poucos exemplos, aproveitando o que aprendeu com itens parecidos.
A terceira, e mais importante, é que o modelo devolve uma probabilidade e um valor em reais, não um simples sim ou não. É isso que permite ordenar as compras e ajustar quantas revisar conforme o tamanho da equipe.
Vale dizer que o modelo não acusa ninguém. Compra atípica não é o mesmo que fraude ou irregularidade: muitas vezes a explicação é um erro de cadastro, um frete caro ou um item diferente. O modelo só organiza o que deve ser olhado primeiro por uma pessoa.
OBJETIVOS
1.3.1 Objetivo geral
Desenvolver e comparar modelos de Machine Learning Clássico para identificar compras públicas com preço atípico e estimar o preço esperado de compras de papel para impressão, usando dados abertos do Compras.gov.br.
1.3.2 Objetivos específicos
1.	Obter e organizar a base de compras pela API do Compras.gov.br, documentando a fonte e o significado de cada variável.
2.	Fazer a Análise Exploratória de Dados (EDA), verificando dados faltantes, valores extremos, unidades de medida e o desbalanceamento das classes.
3.	Criar uma medida de preço comparável entre compras diferentes (reais por quilo de papel) e a regra de compra atípica.
4.	Preparar os dados e criar novas colunas que descrevam a compra sem usar o preço.
5.	Treinar e ajustar dois modelos de classificação (Regressão Logística e Random Forest), priorizando F1 e Recall.
6.	Comparar quatro modelos de regressão para estimar o preço esperado.
7.	Verificar quais variáveis mais pesam na previsão.
8.	Juntar as duas etapas no valor em risco e medir o ganho em relação a escolher compras ao acaso.
9.	Aplicar o K-Means para identificar perfis de compra.
10.	Discutir as limitações e os riscos de usar esse modelo na prática.
1.4 FUNDAMENTAÇÃO TEÓRICA
1.4.1 Revisão do domínio: compras públicas e preço de referência
a) Conceitos-chave
Pesquisa de preços. É o levantamento que o órgão faz antes de comprar, para estimar quanto o item deve custar. No governo federal ela segue a Instrução Normativa SEGES/ME nº 65/2021, que manda usar pelo menos três preços, tirados de sistemas oficiais, contratações anteriores, sites ou fornecedores, e calcular a média, a mediana ou o menor valor (BRASIL, 2021b).
Sobrepreço e superfaturamento. A Lei 14.133/2021 define sobrepreço como o preço "orçado para licitação ou contratado em valor expressivamente superior aos preços referenciais de mercado" (BRASIL, 2021a, art. 6º, LVI). Superfaturamento é outra coisa: é o dano efetivo ao patrimônio público, por exemplo quando se paga por quantidade maior que a entregue (art. 6º, LVII). Este projeto trabalha só com preço, então fala no máximo de indício de sobrepreço, nunca de superfaturamento.
Compra atípica. É o termo que usamos para a compra cujo preço foge do padrão de itens comparáveis. É um sinal estatístico que pede revisão, não uma conclusão. Seguindo o guia do projeto, evitamos a palavra fraude.
CATMAT e PDM. O CATMAT é o catálogo de materiais do governo federal. Cada item tem um código, e itens parecidos são agrupados num PDM (Padrão Descritivo de Material). O PDM 19746, "papel para impressão formatado", reúne os papéis sulfite, offset, couchê, reciclado e texturizado, em vários tamanhos e gramaturas.
Unidade de fornecimento. É como o item é vendido: por folha, por pacote ou por resma. A capacidade diz quantas folhas há na embalagem. É aqui que nasce boa parte dos preços estranhos da base.
Modalidade e forma de compra. Pregão é a disputa aberta entre fornecedores. Dispensa e inexigibilidade são contratações diretas, sem disputa aberta. O Sistema de Registro de Preços (SRP) é quando o órgão registra um preço numa ata e compra aos poucos, ao longo do tempo.
b) Estatísticas e fatos
⦁	As compras governamentais equivalem em média a cerca de 12,5% do PIB do Brasil (RIBEIRO; INÁCIO JÚNIOR, 2019).
⦁	A ferramenta Alice, da CGU, analisou mais de 190 mil processos de compra em 2023 e mais de 161 mil em 2024. Em 2024, as ações preventivas geraram R$ 1,25 bilhão em benefícios financeiros (BRASIL, 2025).
⦁	Na nossa base, 13,3% das 1.350 compras de papel ficaram acima do limite de preço atípico do seu grupo.
⦁	Das 110 compras cadastradas por folha, 51,8% são atípicas, contra 9,9% das cadastradas por embalagem. Isso mostra o peso do erro de unidade.
⦁	A resma de papel A4 sulfite de 75 g/m² (603 compras na base) teve preço mediano de R$ 22,07, com metade das compras entre R$ 19,70 e R$ 25,38.
c) Soluções sem Machine Learning e suas limitações
A pesquisa de preços da IN 65/2021 é a solução oficial e funciona bem quando o servidor encontra preços parecidos. O problema é que ela não define o que é um valor "excessivamente elevado", depende do cuidado de cada servidor e usa bases que já trazem erros de unidade.
O Painel de Preços do governo federal ajuda a encontrar preços de outras compras, mas é uma ferramenta de consulta: alguém precisa ir até ela e saber o que procurar. Ela não aponta sozinha as compras fora do padrão.
As regras fixas, como "revisar tudo acima de X% da mediana", são fáceis de explicar, mas usam um único corte para itens muito diferentes e não levam em conta a quantidade comprada, a embalagem ou a região. A ferramenta Alice, da CGU, vai além e combina regras com mineração de texto, mas se concentra em editais e processos, e não publica um preço esperado para cada compra. É nesse espaço que o nosso modelo contribui: uma referência de preço aprendida com os dados e uma fila de revisão ordenada por dinheiro em risco.
1.4.2 Referencial teórico em Machine Learning
Modelo A: Regressão Logística
A Regressão Logística soma as características da compra, cada uma multiplicada por um peso, e passa o resultado por uma função que aperta tudo entre 0 e 1. O número que sai é a probabilidade de a compra ser atípica. Os pesos são aprendidos com os exemplos de treino. Apesar do nome, é um modelo de classificação: neste projeto, resolve o problema binário de dizer se a compra é atípica ou não.
A grande vantagem é que dá para ler os pesos: peso positivo quer dizer que a característica aumenta o risco, e peso negativo, que diminui. Na administração pública isso importa, porque o auditor precisa justificar por que escolheu revisar uma compra. A desvantagem é que ela separa os grupos com uma linha reta e não percebe sozinha combinações entre variáveis.
Árvore de Decisão
A Árvore de Decisão separa as compras com perguntas de sim ou não, escolhendo em cada passo a pergunta que melhor divide as típicas das atípicas. O resultado é uma regra legível, como "se foi cadastrada por folha e o total comprado é pequeno, o risco é alto". Ela pega combinações naturalmente, mas é instável e, se crescer sem limite, decora os exemplos. É a peça básica do modelo seguinte.
Modelo B: Random Forest (modelo de conjunto)
O Random Forest treina centenas de árvores em vez de uma. Cada árvore recebe um sorteio diferente das compras e, a cada pergunta, só pode olhar algumas variáveis sorteadas. Como cada árvore erra de um jeito diferente, os erros tendem a se anular quando todas votam juntas, e o conjunto fica muito mais estável que uma árvore sozinha (BREIMAN, 2001). Ele também informa quais variáveis mais pesaram nas decisões. O preço é perder a explicação simples. O mesmo método resolve problemas de classificação (aqui, a compra atípica) e de regressão (aqui, o preço esperado).
Modelos de regressão
A Regressão Linear Simples usa uma só variável (aqui, a gramatura) e serve de linha de base. A Linear Múltipla traça a reta que passa mais perto de todos os pontos usando todas as variáveis. A Polinomial acrescenta termos ao quadrado e produtos entre variáveis, permitindo curvas; como isso cria muitos termos, usamos a regularização Ridge, que segura os pesos para o modelo não exagerar. O Random Forest Regressor usa a mesma ideia das centenas de árvores, tirando a média em vez de votar.
Como o preço por quilo é muito assimétrico, os modelos aprendem o logaritmo do preço e depois voltamos para reais. Isso evita que poucas compras caríssimas dominem o ajuste.
K-Means
O K-Means não tem alvo. Ele coloca cada compra no grupo cujo centro está mais perto e vai ajustando os centros até estabilizar (MACQUEEN, 1967). O número de grupos, k, é escolhido pelo coeficiente de silhueta, que mede se cada ponto está mais perto do seu grupo do que dos outros.
Métricas
Na classificação, a Precision diz, entre as compras que o modelo apontou como atípicas, quantas realmente eram. O Recall diz, entre as compras atípicas de verdade, quantas o modelo pegou. O F1 junta os dois numa nota só. O ROC-AUC mede se o modelo coloca as compras na ordem certa de risco, sem depender do corte escolhido.
Na regressão, o MAE é o erro médio em reais por quilo. O RMSE também mede erro, mas pesa muito mais os erros grandes. O MAPE é o erro médio em porcentagem. O R² diz quanto da variação do preço o modelo consegue explicar. Incluímos também o erro mediano, que não é afetado por um único erro gigante.
1.5 METODOLOGIA E DESENVOLVIMENTO
Todo o projeto foi feito em Python, no Google Colab, com a biblioteca scikit-learn (PEDREGOSA et al., 2011). A semente aleatória foi fixada em 42 em todas as etapas, então os resultados se repetem sempre que o notebook é executado com o mesmo arquivo de dados.
1.5.1 Análise Exploratória de Dados (EDA)
Descrição. A base tem 1.350 linhas e 18 colunas. São 14 colunas vindas da API de preços (identificador, código do item, data, modalidade, forma, critério de julgamento, quantidade, preço unitário, unidade de fornecimento, capacidade da embalagem, UF, esfera, UASG e CNPJ do fornecedor) e 4 do catálogo (tipo de papel, tamanho, gramatura e cor). Três colunas são numéricas (quantidade, preço e capacidade), duas são códigos inteiros e as demais são texto. As compras vêm de 646 unidades compradoras, nos 27 estados, das três esferas (593 federais, 399 estaduais e 358 municipais), e envolvem 124 itens diferentes do catálogo e 447 fornecedores.
Valores ausentes. Só a coluna de critério de julgamento tem dados faltando: 18 valores (1,3% da base). Essa coluna não foi usada no modelo, porque traz códigos sem documentação clara e porque a modalidade já descreve a forma de disputa.
Linhas duplicadas. Nenhuma. Cada item de compra tem um identificador único. Há linhas muito parecidas, como vários lotes do mesmo papel na mesma compra, mas são itens diferentes e foram mantidas.
O preço bruto não é comparável. O mesmo papel aparece vendido por folha (110 compras) ou por embalagem (1.240 compras), e as embalagens vão de 10 a 500 folhas. Além disso, uma folha A3 tem o dobro de papel de uma A4. Por isso convertemos tudo para reais por quilo de papel: área da folha × gramatura × folhas por unidade dá os quilos de cada unidade comprada, e o preço unitário dividido por esses quilos dá o preço por kg. Uma resma A4 de 75 g/m² pesa 2,34 kg; se custou R$ 20,00, o preço é R$ 8,55 por kg.
Valores extremos. O preço por kg tem mediana de R$ 10,90, mas vai de R$ 0,05 a R$ 447.990. Nenhum papel custa isso: esses extremos são quase sempre erros de unidade, como uma resma cadastrada como se fosse uma folha. Não removemos esses valores, porque identificá-los é justamente o objetivo do projeto. Em 94 compras o preço por kg é mais de 10 vezes a mediana do seu grupo.
Grupos de comparação. Couchê e papel texturizado são naturalmente mais caros por quilo que o sulfite. Por isso cada compra só é comparada com compras do mesmo tipo de papel. O tipo "permanente" tinha só 2 compras e foi juntado com o texturizado no grupo de papéis especiais.
Regra de compra atípica. A compra é atípica quando o preço por kg passa de Q3 + 1,5 × IQR do seu grupo, a mesma regra que marca os pontos extremos de um boxplot (TUKEY, 1977). Os quartis foram calculados só com as compras de treino. O resultado está na Tabela 1.
Tabela 1: Limites de preço por grupo de comparação (R$ por kg, compras de treino)
Grupo	Q1	Mediana	Q3	Limite atípico
Sulfite	8,47	10,26	14,69	24,01
Reciclado	9,60	10,68	13,90	20,34
Offset	9,65	11,54	33,20	68,54
Couchê	11,26	19,43	34,38	69,05
Especial (texturizado e permanente)	20,15	32,96	47,57	88,69
Fonte: elaborado pelos autores.
Desbalanceamento. 180 das 1.350 compras (13,3%) são atípicas. Isso torna a acurácia enganosa: um modelo que diz que nenhuma compra é atípica acerta 86,7% e não serve para nada.
Figura 1: Preço por kg por tipo de papel e taxa de compras atípicas por unidade
 
Fonte: elaborado pelos autores.
O gráfico da esquerda, em escala logarítmica, mostra que a maioria das compras fica numa faixa estreita, mas há pontos centenas de vezes acima dela. O da direita mostra a descoberta mais forte da análise: entre as compras cadastradas por folha, 51,8% são atípicas, contra 9,9% das cadastradas por embalagem. Todas as 10 compras de "embalagem com 10 folhas" são atípicas, o que indica caixas com 10 resmas registradas como pacotes de 10 folhas. Também chama atenção a esfera: 23,7% das compras municipais são atípicas, contra 11,0% das federais e 7,5% das estaduais.
1.5.2 Pré-processamento e engenharia de atributos
Divisão dos dados. Separamos 80% das compras para treino (1.080) e 20% para teste (270). O teste funciona como uma prova que o modelo nunca viu: só é usado no final, para medir se ele aprendeu de verdade ou só decorou. A divisão foi estratificada pelo tipo de papel e foi feita antes de criar o alvo, para que o próprio limite de preço atípico fosse aprendido só com o treino. Ficaram 13,5% de atípicas no treino e 12,6% no teste. Para ajustar os modelos, usamos validação cruzada em 5 partes, sempre dentro do treino: em cada rodada, 4 partes treinam o modelo e 1 parte o valida. Na prática, as 1.350 compras ficam divididas em 64% para treino (864), 16% para validação (216) e 20% para teste (270). Essa divisão em três conjuntos é importante porque a validação serve para escolher os hiperparâmetros sem tocar no teste; assim, a nota final no teste não é otimista e mostra como o modelo se comportaria com compras novas.
Dados ausentes. Não houve preenchimento. A única coluna com faltantes (critério de julgamento) foi deixada de fora do modelo, pelos motivos explicados na EDA. As colunas do catálogo foram completadas para todos os 124 itens, então não sobrou nenhum faltante nas variáveis usadas.
Sem preço na classificação. O alvo da classificação é feito a partir do preço. Se o preço entrasse como variável, o modelo estaria "prevendo" a resposta olhando a resposta. Por isso a classificação usa só o que se sabe da compra antes do preço final: o que se compra, quanto, de que forma e por quem.
Codificação das categorias. Tipo de papel, modalidade, esfera e região são rótulos, não quantidades. Usamos One-Hot, que transforma cada categoria numa coluna de sim ou não. A inexigibilidade tinha uma única compra e foi juntada com a dispensa no grupo "contratação direta".
Padronização. Usamos o StandardScaler, que coloca as colunas numéricas na mesma escala, com média 0 e desvio 1. Sem isso, as folhas por embalagem (até 500) dominariam a área da folha (menos de 1 m²). Escolhemos o StandardScaler em vez do MinMaxScaler porque há valores extremos, que fariam o MinMaxScaler espremer quase todas as compras perto do zero. A padronização fica dentro de um Pipeline, para ser calculada só com os dados de treino.
Engenharia de atributos. Criamos estas colunas:
⦁	preco_kg: preço por quilo de papel, usado para criar o alvo e como alvo da regressão.
⦁	area_m2 e gramatura_g: área da folha e gramatura, tiradas do texto do catálogo.
⦁	folhas_unid: folhas por unidade comprada (1 quando a unidade é a folha).
⦁	log_quantidade e log_kg_total: tamanho da compra em escala log, porque a quantidade vai de 1 a mais de 600 mil.
⦁	unidade_folha: 1 se o item foi cadastrado por folha.
⦁	formato_a4, cor_branca e registro_precos: indicadores de papel A4, papel branco e compra por ata de registro de preços.
⦁	regiao e mes: região do país, a partir da UF, e mês da compra.
1.5.3 Implementação, treinamento e otimização
Os modelos foram treinados com a biblioteca scikit-learn, cada um dentro de um Pipeline junto com a padronização e o One-Hot. Nos dois modelos de classificação usamos o ajuste de peso das classes (class_weight = balanced), que faz o erro sobre as compras atípicas pesar mais. Sem isso o modelo percebe que dizer "nenhuma é atípica" já acerta 87% e para de arriscar.
Os hiperparâmetros foram ajustados com GridSearchCV, que testa várias combinações e escolhe a melhor em validação cruzada de 5 partes, só com dados de treino. Na classificação o critério foi o F1; na regressão, o menor erro médio absoluto.
Quadro 1: Ajuste dos hiperparâmetros
Modelo	O que foi testado	Escolhido	Validação
Regressão Logística	C = 0,01; 0,1; 1; 10	C = 10	F1 = 0,512
Random Forest (300 árvores)	profundidade 4, 8 ou sem limite; folha mínima 1, 5 ou 10	sem limite; folha 5	F1 = 0,663
Polinomial grau 2 (Ridge)	alpha = 1; 10; 100; 1000	alpha = 10	MAE = R$ 2,98/kg
Random Forest Regressor (400 árvores)	folha mínima 1, 3, 5 ou 10; variáveis por divisão 100% ou 50%	folha 1; 50%	MAE = R$ 2,55/kg
Fonte: elaborado pelos autores.
Na regressão, o modelo foi treinado só com as compras típicas do treino (934 compras) e avaliado nas compras típicas do teste (236). Se deixássemos as atípicas no treino, o modelo aprenderia que preço absurdo é normal. É o mesmo raciocínio da IN 65/2021, que manda descartar os valores discrepantes antes de calcular o preço de referência.
1.5.4 Avaliação e análise de resultados
Por que essas métricas
Neste problema os dois erros têm custos diferentes. Deixar passar uma compra atípica pode significar dinheiro público pago a mais ou uma referência de preço contaminada. Apontar uma compra normal custa o tempo de um servidor. Por isso olhamos o Recall, que mede quantas atípicas o modelo pega, mas escolhemos o modelo pelo F1, porque uma equipe de controle também não pode ser afogada em alarmes falsos. O ROC-AUC foi usado para ver qual modelo ordena melhor as compras por risco, que é o que importa para a fila. A acurácia aparece só para comparação.
Comparação dos modelos de classificação
Tabela 2: Resultado no conjunto de teste (270 compras, 34 atípicas)
Modelo	Acurácia	Precision	Recall	F1	ROC-AUC	Apontadas
Random Forest	88,9%	55,3%	61,8%	0,583	0,917	14,1%
Regressão Logística	80,7%	36,8%	73,5%	0,490	0,874	25,2%
Fonte: elaborado pelos autores.
O Random Forest ganhou no F1 e no ROC-AUC, por isso foi o modelo escolhido. A Regressão Logística teve Recall maior porque marca muito mais compras como atípicas: uma em cada quatro, contra uma em cada sete do Random Forest. Quem marca mais pega mais atípicas, mas também gera mais alarmes falsos. O F1 equilibra essas duas coisas, e nele o Random Forest foi melhor.
A vantagem do Random Forest vem de ele pegar combinações entre variáveis. Uma compra pequena não é suspeita por si só, nem uma compra cadastrada por folha; o risco aparece quando as duas coisas vêm juntas. A Logística, em compensação, é mais fácil de explicar.
Figura 2: Curva ROC e matrizes de confusão no conjunto de teste
 
Fonte: elaborado pelos autores.
Das 34 compras atípicas do teste, o Random Forest pegou 21 e deixou passar 13, apontando 17 compras normais como atípicas. A Logística pegou 25 e deixou passar 9, mas apontou 43 compras normais. As duas curvas ROC ficam bem acima da linha tracejada, que seria escolher ao acaso, e a do Random Forest fica acima da Logística em quase toda a extensão.
Variáveis que mais pesam
Figura 3: Importância das variáveis no Random Forest
 
Fonte: elaborado pelos autores.
As quatro variáveis mais importantes são os quilos totais comprados (0,289), as folhas por embalagem (0,153), a quantidade (0,142) e o cadastro por folha (0,062). Todas descrevem como a quantidade e a unidade foram registradas. A Regressão Logística conta a mesma história pelos sinais: quilos totais têm o maior peso negativo (−3,31) e quantidade tem peso positivo (1,73). Ou seja, o risco é alto quando a quantidade registrada é grande mas o papel correspondente pesa pouco, que é exatamente o que acontece quando uma resma é cadastrada como folha.
Esfera municipal (0,046) e região aparecem logo depois. Tipo de papel e modalidade pesam pouco. Em resumo, o sinal mais forte de preço atípico não é quem compra nem como, é a unidade de fornecimento não bater com o preço.
Regressão: estimando o preço esperado
Tabela 3: Resultado dos modelos de regressão (compras típicas de teste, R$ por kg)
Modelo	MAE	RMSE	Erro mediano	MAPE	R²
Random Forest Regressor	2,78	6,20	0,87	22,3%	0,580
Linear Múltipla	3,98	7,83	1,99	34,0%	0,332
Linear Simples (gramatura)	4,89	9,38	2,01	45,8%	0,039
Polinomial grau 2 (Ridge)	6,42	43,50	1,62	42,9%	−19,652
Fonte: elaborado pelos autores.
O Random Forest Regressor foi o melhor em todas as medidas: errou em média R$ 2,78 por kg, com erro mediano de R$ 0,87 por kg, e explicou 58% da variação do preço. A Linear Múltipla explicou 33% e a Linear Simples, só com a gramatura, quase nada (4%), o que mostra que o preço depende de várias coisas ao mesmo tempo.
A Polinomial merece atenção. Na validação cruzada ela foi razoável (MAE de R$ 2,98 por kg) e no teste teve o segundo menor erro mediano. Mas numa única compra rara, um papel texturizado cadastrado por folha, os termos ao quadrado extrapolaram e ela previu R$ 686 por kg para um papel que custou R$ 25. Esse único ponto derrubou o R² para −19,65; sem ele, o R² seria 0,515. É o risco clássico da regressão polinomial: fora da faixa dos dados de treino, as curvas disparam.
Figura 4: Preço real versus preço esperado (Random Forest Regressor)
 
Fonte: elaborado pelos autores.
Os pontos se concentram perto da diagonal: 76% das previsões erram menos de 25% para mais ou para menos, e 90% erram menos de 50%. Na prática, para a resma A4 de 75 g/m², o modelo estima R$ 22,90 contra R$ 23,14 do preço real mediano nas compras de teste.
Juntando as duas etapas: o valor em risco
Com o preço esperado, calculamos para cada compra o sobrepreço estimado: a diferença entre o preço pago e o esperado, quando positiva, vezes a quantidade. O valor em risco é a probabilidade de a compra ser atípica vezes esse sobrepreço. Ordenando por esse valor, temos a fila de revisão.
Tabela 4: Fila de revisão na base de teste
Indicador	Valor
Compras na base de teste	270
Valor em risco estimado	R$ 1,59 milhão
Parcela do valor em risco nos 10% do topo (27 compras)	97,9%
Compras atípicas nos 10% do topo	81,5%
Compras atípicas na base de teste toda	12,6%
Compras do topo com preço 5 vezes ou mais acima do esperado	20 de 27
Fonte: elaborado pelos autores.
Entre as 27 compras do topo da fila, 81,5% são atípicas, 6,5 vezes mais que na base toda, e elas concentram quase todo o valor em risco. Há um cuidado importante aqui: o sobrepreço usa o preço pago, que também define o rótulo, então era esperado que o topo tivesse muitas atípicas. O ganho real da fila está em ordenar pelo dinheiro envolvido.
Também é preciso ler esse valor com cuidado. Das 27 compras do topo, 20 têm preço 5 vezes ou mais acima do esperado, o que quase sempre é erro de unidade no cadastro, e não dinheiro pago a mais. Só 3 ficam entre 1,5 e 5 vezes o esperado, que é a faixa onde está o candidato a sobrepreço real. Por isso o R$ 1,59 milhão não deve ser lido como prejuízo, e sim como o tamanho da distorção que chega à base de preços.
Perfis de compra com K-Means
Figura 5: Grupos encontrados pelo K-Means
 
Fonte: elaborado pelos autores.
O K-Means usou três medidas: o preço relativo à mediana do grupo, os quilos totais comprados e o cadastro por folha. A melhor silhueta foi com dois grupos (0,681). O primeiro reúne 120 compras, 91% cadastradas por folha, com preço mediano 8,4 vezes acima do seu grupo e 57% de atípicas. O segundo reúne as outras 1.230 compras, com preço mediano igual ao do grupo e 9% de atípicas. O agrupamento confirma, sem usar o rótulo, o que a classificação encontrou: existe um bloco de compras com cadastro de unidade problemático.
Análise crítica
O Random Forest foi o melhor modelo de classificação e de regressão porque o preço atípico depende de combinações entre variáveis, e ele pega isso sozinho. A vantagem sobre os modelos lineares foi clara na regressão (R² de 0,58 contra 0,33) e moderada na classificação (F1 de 0,58 contra 0,49). A Regressão Logística tem a vantagem de ser explicável, o que num órgão de controle também pesa. Há, portanto, uma compensação entre desempenho e interpretabilidade: o Random Forest acerta mais (F1 de 0,583 contra 0,490), mas só informa a importância geral de cada variável, enquanto a Logística mostra o sinal e o peso de cada uma, o que ajuda o auditor a justificar por que uma compra foi escolhida para revisão.
O resultado mais útil do projeto não era o esperado. Imaginávamos encontrar diferenças de preço por região ou por modalidade, mas o fator dominante foi a qualidade do cadastro da unidade de fornecimento. Isso muda a recomendação: antes de caçar sobrepreço, vale limpar a base.
1.5.5 Conclusão, produto final e cronograma
Resposta ao problema
A pergunta do projeto era quais compras merecem revisão por apresentarem preço fora do padrão de itens comparáveis. A resposta é uma fila de revisão ordenada por valor em risco, em que os 10% do topo concentram 97,9% do valor em risco e têm 6,5 vezes mais compras atípicas que a média. O modelo de regressão ainda entrega um preço esperado para cada compra, que pode servir de referência na pesquisa de preços.
A recomendação prática tem duas partes. Na entrada dos dados, validar a unidade de fornecimento e a capacidade da embalagem sempre que o preço por kg sair muito do padrão, o que limparia a base que todo o governo usa como referência. Na revisão, mandar para análise humana as compras do topo da fila que não forem erro de unidade, que são as candidatas a sobrepreço real. O modelo não substitui o auditor nem acusa ninguém; ele só organiza o que deve ser olhado primeiro.
Limitações
⦁	A base cobre só cinco meses (agosto a dezembro de 2025) e um tipo de item. Para outros itens seria preciso treinar de novo.
⦁	Usamos as 1.350 compras mais recentes de 2025 devolvidas pela API, e não o ano todo. O notebook tem uma função para baixar o ano completo.
⦁	O rótulo vem de uma regra estatística (Q3 + 1,5 × IQR). O modelo aprende essa regra, e compra atípica não é o mesmo que compra irregular. Mudar o corte muda os resultados.
⦁	O sobrepreço estimado está inflado por erros de cadastro e não é prejuízo real.
⦁	A base não tem frete, prazo de entrega nem marca exigida, que também explicam diferenças de preço.
⦁	O modelo mostra associação, não causa. Nenhuma variável prova que houve irregularidade.
1.5.5.1 Entregáveis
1.	Documento final: este relatório.
2.	Código-fonte: notebook do Google Colab (Projeto_ML_Compras_Gov.ipynb) com a execução completa e o arquivo de dados compras_papel_2025.csv, disponíveis em repositório no GitHub: ____________________.
3.	Apresentação: slides para a defesa oral.
4.	Especificações técnicas: Python 3 com pandas, numpy, scikit-learn, matplotlib e seaborn. Roda no Google Colab, sem GPU, em poucos minutos. Semente aleatória 42.
Cronograma
Quadro 2: Cronograma do projeto
Etapa	Atividade	Semana
1	Escolha do tema, coleta dos dados pela API e pesquisa	1
2	Análise exploratória, normalização do preço e regra de atipicidade	2
3	Preparação dos dados e criação de atributos	3
4	Treino e ajuste dos modelos de classificação	4
5	Regressão, fila de revisão e K-Means	5
6	Relatório final e apresentação	6
Fonte: elaborado pelos autores.
REFERÊNCIAS
BRASIL. Lei nº 14.133, de 1º de abril de 2021. Lei de Licitações e Contratos Administrativos. Diário Oficial da União, Brasília, DF, 1 abr. 2021a. Disponível em: https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/l14133.htm. Acesso em: 25 set. 2026.
BRASIL. Ministério da Economia. Secretaria de Gestão. Instrução Normativa SEGES/ME nº 65, de 7 de julho de 2021. Dispõe sobre o procedimento administrativo para a realização de pesquisa de preços. Diário Oficial da União, Brasília, DF, 2021b. Disponível em: https://www.gov.br/compras/pt-br/acesso-a-informacao/legislacao/instrucoes-normativas/instrucao-normativa-seges-me-no-65-de-7-de-julho-de-2021. Acesso em: 25 set. 2026.
BRASIL. Controladoria-Geral da União. Alice: Analisador de Licitações, Contratos e Editais. Brasília: CGU, 2025. Disponível em: https://www.gov.br/cgu/pt-br/assuntos/auditoria-e-fiscalizacao/alice. Acesso em: 25 set. 2026.
BREIMAN, Leo. Random Forests. Machine Learning, v. 45, n. 1, p. 5-32, 2001.
COMPRAS.GOV.BR. API de Dados Abertos: módulos de Pesquisa de Preço e de Material. Brasília, 2026a. Disponível em: https://dadosabertos.compras.gov.br/swagger-ui/index.html. Acesso em: 25 set. 2026.
COMPRAS.GOV.BR. Compras Públicas em Dados Abertos. Brasília, 2026b. Disponível em: https://www.gov.br/compras/pt-br/cidadao/compras-publicas-dados-abertos. Acesso em: 25 set. 2026.
MACQUEEN, James. Some methods for classification and analysis of multivariate observations. In: BERKELEY SYMPOSIUM ON MATHEMATICAL STATISTICS AND PROBABILITY, 5., 1967, Berkeley. Proceedings. Berkeley: University of California Press, 1967. v. 1, p. 281-297.
PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.
RIBEIRO, Cássio Garcia; INÁCIO JÚNIOR, Edmundo. O mercado de compras governamentais brasileiro (2006-2017): mensuração e análise. Brasília: Ipea, 2019. (Texto para Discussão, n. 2476). Disponível em: http://repositorio.ipea.gov.br/bitstream/11058/9315/1/td_2476.pdf. Acesso em: 25 set. 2026.
TUKEY, John W. Exploratory Data Analysis. Reading: Addison-Wesley, 1977.
