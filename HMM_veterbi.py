
class HMM(object):
    def __init__(self):
        import os
        
        #模型训练结果
        self.model_file = './data/C3_hmm_model.pkl'
        
        #状态值集合
        self.state_list = ['B', 'M', 'E', 'S']
        #参数加载，用于判断是否需要加载model_file
        self.load_para = False
    
    #用于加载已计算的中间结果，当需要重新加载时，初始化清空中间结果
    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        
        else:
            #状态转移矩阵概率
            self.A_dic = {}
            #发射矩阵概率
            self.B_dic = {}
            #状态的初始概率
            self.Pi_dic = {}
            
            self.load_para = False
    
    def train(self, path):
        #重置概率矩阵
        self.try_load_model(False)
        #统计状态出现的次数，求 p(o)
        Count_dic = {}
        
        #初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s:0 for s in self.state_list}
                self.B_dic[state] = {}
                self.Pi_dic[state] = 0.0
                
                Count_dic[state] = 0
        
        #状态标签
        def markLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            
            else:
                out_text += ['B'] + ['M']*(len(text) - 2) + ['E']
            
            return out_text
        
        init_parameters()
        line_num = 0
        
        #观察者集合，主要是字和标点等
        words = set()
        
        with open(path, encoding = 'utf-8') as f:
            for line in f:
                line_num += 1
                
                line = line.strip()
                if not line:
                    continue
                
                word_list = [i for i in line if i != ' ']
                words |= set(word_list) #更新字的集合
                
                linelist = line.split()
                line_state = []
                for w in linelist:
                    line_state.extend(markLabel(w))
                
                assert len(word_list) == len(line_state)
                
                for k, v in enumerate(line_state):
                    Count_dic[v] += 1
                    
                    if k == 0:
                        self.Pi_dic[v] += 1  #每个句子的第一个字，用于计算初始状态概率
                    else:
                        self.A_dic[line_state[k-1]][v] += 1  #转移概率
                        self.B_dic[line_state[k]][word_list[k]] = \
                        self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0  #发射概率
                        
        self.Pi_dic = {k: v*1.0/line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1/Count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}
        #拉普拉斯平滑（+1）
        self.B_dic = {k: {k1: (v1+1)/Count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dic.items()}
        
        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)
        
        return self
    
    def veterbi(self, text, states, start_p, trans_p, emit_p):
        #状态及其概率的序列
        V = [{}]
        #终止于某状态的序列
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            
            #检验训练的发射矩阵中是否有该字
            neverSeen = text[t] not in emit_p['B'].keys() and \
                        text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and \
                        text[t] not in emit_p['S'].keys()
            
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0 #设置未知字单独成词
                #对目前的每个状态，遍历上个位置所有状态转移再发射后的概率的最大值，及其对应的上一个状态
                (prob, state) = max(
                    [(V[t - 1][y_pre] * trans_p[y_pre].get(y, 0) * emitP, y_pre)
                     for y_pre in states if V[t - 1][y_pre] > 0])
                V[t][y] = prob
                #当前状态与该状态对应的最优之前状态序列的拼接
                newpath[y] = path[state] + [y]
            
            path = newpath
        
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ['E', 'M']])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        
        return (prob, path[state])
    
    def cut(self, text):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        
        prob, pos_list = self.veterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin:i+1]
                next = i+1
            elif pos == 'S':
                yield char
                next = i + 1
        
        if next < len(text):
            yield text[next:]
    