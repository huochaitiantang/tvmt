if __name__ == '__main__':
    model_runtime = {}
    with open('result', 'r') as f:
        lines = f.readlines()
        model = ''
        for line in lines:
            if len(line) > len('model name') and line[:len('model name')] == 'model name':
                model = line.split(' ')[-1].replace('\n', '')
            elif len(line) > len('Total_time') and line[:len('Total_time')] == 'Total_time':
                model_runtime[model] = line.replace(' ', '').replace('-', '')[len('Total_time'):-1]
    for key, value in model_runtime.items():
        print(key, value)
