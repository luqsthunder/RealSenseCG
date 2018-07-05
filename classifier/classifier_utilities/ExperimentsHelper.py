import matplotlib.pyplot as plt

class ExperimentCnn3D:
	def __init__(self, cnn_layers, filters_amount, denses_layers, accuracies, conf_mat):
		self.accuracies = accuracies
		self.denses_layers = denses_layers
		self.cnn_layers = cnn_layers
		self.filters_amount = filters_amount
		self.conf_mat = conf_mat
		
		
file_path = 'C:/Users/lucas/Downloads/Telegram Desktop/cnn3Doutput.txt'
accuracies = []
file = open(file_path, 'r')

experiments = []

for exps in range(0, 702):
	line = ''
	while('cnn3D' not in line):
		line = file.readline(102)
	
	curr_content = [line]	
	for line_idx in range(0 , 15):
		c = file.readline(102)
		curr_content.append(c)
	
	# extrai matriz de confusão
	conf_mat = []
	for idx_mat in range(4, 14):
		f0 = curr_content[idx_mat].rfind('[')
		f1 = curr_content[idx_mat].find(']')
		strvec = [l.replace(' ', '') for l in curr_content[idx_mat][f0+1:f1].split('. ')]
		floatvec = [float(l) for l in strvec]
		conf_mat.append(floatvec)
	
	# extrai configurações dos filtros CNN
	f0 = curr_content[0].rfind('[')
	f1 = curr_content[0].find(']')
	cnn_layer = [int(l) for l in curr_content[0][f0+1:f1].split(', ')]
	
	# extrai configurações da quantidade dos filtros
	f0 = curr_content[1].rfind(':')
	f1 = curr_content[1].rfind('\n')
	filters_amount = int(curr_content[1][f0+1:f1])
	
	# extrai configurações das camadas densas
	f0 = curr_content[2].rfind('[')
	f1 = curr_content[2].rfind(']')
	denses_layers = [int(l) for l in curr_content[2][f0+1:f1].split(', ')]
	
	# extrai configurações de todas as accuracias
	f0 = curr_content[15].rfind('[')
	f1 = curr_content[15].rfind(']')
	accuracies = [float(l) for l in curr_content[15][f0+1:f1].split(', ')]
	
	print('exp saving {}, cnn {}, amount filters {}, denses {}\n'.format(exps, cnn_layer, filters_amount, denses_layers))
	experiments.append(ExperimentCnn3D(cnn_layer, filters_amount, denses_layers, accuracies, conf_mat))

epochs = [max(l.accuracies) for l in experiments]
epochs_last = [l.accuracies[4] for l in experiments]

plt.title('Best Accuracies for each CNN3D Architecture')
plt.xlabel('Accuracy')
plt.hist(epochs, color=['gray'])
plt.savefig('best Acc Cnn3d.svg')

plt.title('Accuracies for last epoch of each CNN3D Architecture')
plt.xlabel('Accuracy')
plt.hist(epochs, color=['gray'])
plt.savefig('Last Epoch Cnn3d.svg')