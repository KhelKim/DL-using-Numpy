import csv
import time
import numpy as np

np.random.seed(1234)


def randomize(): np.random.seed(time.time())


# hyperparameter ################
_rnd_mean = 0
_rnd_std = 0.0030
_learning_rate = 0.001
# hyperparameter ################


# 함수 첫 번째 깊이, 함수 1: 실험용 메인 함수
def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


# 함수 두 번째 깊이, 함수 1: 데이터 적재 함수
def load_abalone_dataset():
    with open("../../data/chap01/abalone.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        # next(csv_reader, None)
        # next(iter 객체, iter 객체에 더 이상 꺼낼 원소가 없을 때 원래 에러가 나지만 이 값을 세팅해놓으면 그 값을 뱉어냄)
        # 내가 받은 csv 파일은 header가 없어 해당 코드를 제거
        rows = [row for row in csv_reader]
    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros(shape=[len(rows), input_cnt+output_cnt])
    for n, row in enumerate(rows):
        if row[0] == "I": data[n, 0] = 1
        elif row[0] == "M": data[n, 1] = 1
        elif row[0] == "F": data[n, 2] = 1
        data[n, 3:] = row[1:]


# 함수 두 번째 깊이, 함수 2: 파라미터 초기화 함수
def init_model():
    # 함수 내부에서 전역 변수에게 접근할 때는 명시적으로 언급해주지 않아도 됨
    # 함수 내부에서 전역 변수에게 변경을 가하거나 새로운 값을 할당할 때는 언급해주어야 함
    global weight, bias
    weight = np.random.normal(loc=_rnd_mean, scale=_rnd_std, size=[input_cnt, output_cnt])
    bias = np.zeros([output_cnt])


# 함수 두 번째 깊이, 함수 3: 학습 및 평가 함수 정의
def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        # epoch
        losses, accs = [], []

        for n in range(step_count):
            # mini batch step
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print(f'Epoch {epoch+1}: loss={np.mean(losses):7.3f},',
                  f'accuracy={np.mean(accs):7.3f}(train), {acc:7.3f}(test)')
            # f{variable:10.3f}
            # float를 소수점 3자리까지 보여준다.
            # float의 전체길이를 10으로 맞춘다.

    final_acc = run_test(test_x, test_y)
    print(f'\nFinal Test: final accuracy = {final_acc:7.3f}')


# 함수 세 번째 깊이, 함수 1: 학습 및 평가 데이터 획득함수 정의
def arrange_data(mb_size):
    global shuffle_map, test_begin_idx
    shuffle_map = np.arange(start=data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count


# 함수 세 번째 깊이, 함수 2: 학습 및 평가 데이터 획득함수 정의
def get_test_data():
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


# 함수 세 번째 깊이, 함수 3: 학습 및 평가 데이터 획득함수 정의
def get_train_data(mb_size, nth):
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]


# 함수 세 번째 깊이, 함수 4: 학습 실행 함수와 평가 실행 함수 정의
def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)
    return loss, accuracy


# 함수 세 번째 깊이, 함수 5: 학습 실행 함수와 평가 실행 함수 정의
def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


# 함수 네 번째 깊이, 함수 1: 단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
def forward_neuralnet(x):
    output = np.matmul(x, weight) + bias
    return output, x


# 함수 네 번째 깊이, 함수 2: 단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)
    weight -= _learning_rate * G_w
    bias -= _learning_rate * G_b


# 함수 네 번째 깊이, 함수 3: 후처리 과정에 대한 순전파 및 역전파 함수 정의
def forward_postproc(output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


# 함수 네 번째 깊이, 함수 4: 후처리 과정에 대한 순전파 및 역전파 함수 정의
def backprop_postproc(G_loss, diff):
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output


# 함수 네 번째 깊이, 함수 5: 후처리 과정에 대한 순전파 및 역전파 함수 정의
def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff
