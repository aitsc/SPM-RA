import pickle

class 还原训练语料:
    def __init__(self,答案地址,训练集地址,词表地址):
        with open(答案地址,'rb') as r:
            self._答案序号_句子d = pickle.load(r)
        with open(词表地址,'rb') as r:
            self._序号_词表d = pickle.load(r)
        self._问题_答案 = []
        with open(训练集地址, 'rb') as r:
            for 一对 in pickle.load(r):
                问题l = 一对['question']
                答案l = [self._答案序号_句子d[i] for i in 一对['answers']]
                self._问题_答案.append([问题l] + 答案l)

    def 输出(self,输出地址,输出所有问题=False,输出原词=False):
        if 输出原词:
            词转换 = lambda t:self._序号_词表d[t]
        else:
            词转换 = lambda t:str(t)
        with open(输出地址,'w',encoding='utf-8') as w:
            if 输出所有问题:
                for ll in self._问题_答案:
                    w.write(' '.join([词转换(i) for i in ll[0]]))
                    w.write('\n')
                for 答案 in self._答案序号_句子d.values():
                    w.write(' '.join([词转换(i) for i in 答案]))
                    w.write('\n')
            else:
                for ll in self._问题_答案:
                    w.write('\t'.join([' '.join([词转换(i) for i in l]) for l in ll]))
                    w.write('\n')

    def 成对输出(self,输出地址,输出原词):
        if 输出原词:
            词转换 = lambda t:self._序号_词表d[t]
        else:
            词转换 = lambda t:str(t)
        with open(输出地址,'w',encoding='utf-8') as w:
            for ll in self._问题_答案:
                问题l = ll[0]
                for 答案l in ll[1:]:
                    w.write(' '.join([词转换(i) for i in 问题l])+'\t'+' '.join([词转换(i) for i in 答案l]))
                    w.write('\n')


if __name__ == '__main__':
    # 还原训练语料('answers','train').输出('train.txt')
    # 还原训练语料('answers','train','vocabulary').输出('train_answers.txt',输出所有问题=True,输出原词=True)
    # 还原训练语料('answers','train','vocabulary').输出('train_answers_number.txt',输出所有问题=True,输出原词=False)
    还原训练语料('answers','train','vocabulary').成对输出('train_QtapA.txt',输出原词=True)