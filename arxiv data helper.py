import pickle
import os
import re
from tqdm import tqdm
import sys
import time
import shutil

class PaperFilter:
    def __init__(self,id_title_author_c1_c2_abstract_year_L):
        self._authorName_paperId_c1_c2_D = self._creatAuthorInfor(id_title_author_c1_c2_abstract_year_L)
        self._id_title_author_c1_c2_abstract_year_D = {i[0]:i[1:] for i in id_title_author_c1_c2_abstract_year_L}
        self._reviewerName_paperId_class_D = {}
        self._manuscriptId_reviewer_class_D = {}
    def filter(self,reviewerNum=1000,reviewerYear_L=[0,2015],reviewerPaperWordNum_L=[100,1000],reviewerPaperNum_L=[50,150],
               manuscriptNum=1000,manuscriptYear_L=[2016,2019],manuscriptWordNum_L=[150,1000],manuscriptReviewerNum_L=[20,100],
               whatClassUsed = 'subject',howClassNumForQual = 10):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if whatClassUsed == 'subject':
            whatClassUsed = 2
        elif whatClassUsed == 'MSC':
            whatClassUsed = 3
        else:
            assert False, '分类错误!'
        reviewerName_paperId_class_D = {}
        manuscriptId_reviewer_class_D = {}
        class_reviewerName_D = {}
        allReviewerPaper_S = set()

        for author,[paperId_D,c1_num_D,c2_num_d] in tqdm(self._authorName_paperId_c1_c2_D.items(),mName+'-筛选审稿人'):
            paperId_class_L = [{},{}]
            if reviewerPaperNum_L[0]>len(paperId_D) or len(paperId_D)>reviewerPaperNum_L[1]:
                continue
            for id in paperId_D.keys():
                p = self._id_title_author_c1_c2_abstract_year_D[id]
                if p[5]<reviewerYear_L[0] or p[5]>reviewerYear_L[1]:
                    continue
                if not p[0] or not p[4] or len(p[0])==0 or len(p[4])==0:
                    continue
                text = p[0] +' '+ p[4]
                if reviewerPaperWordNum_L[0]<=len(text.split())<=reviewerPaperWordNum_L[1]:
                    for i in p[whatClassUsed]:
                        if i in paperId_class_L[1]:
                            paperId_class_L[1][i] += 1
                        else:
                            paperId_class_L[1][i] = 1
                    paperId_class_L[0][id] = None
            if reviewerPaperNum_L[0]>len(paperId_class_L[0]) or len(paperId_class_L[0])>reviewerPaperNum_L[1]:
                continue
            if max(list(paperId_class_L[1].values())) < howClassNumForQual:
                continue
            for c,i in paperId_class_L[1].items():
                if i>=howClassNumForQual:
                    if c in class_reviewerName_D:
                        class_reviewerName_D[c][author] = i
                    else:
                        class_reviewerName_D[c] = {author:i}
            reviewerName_paperId_class_D[author] = paperId_class_L
            allReviewerPaper_S |= set(paperId_class_L[0].keys())
            if len(reviewerName_paperId_class_D)>=reviewerNum:
                break
        assert reviewerName_paperId_class_D,'无满足条件的审稿人!'
        for paperId,paperInfor_L in tqdm(self._id_title_author_c1_c2_abstract_year_D.items(),mName+'-筛选稿件'):
            reviewer_class_L = [{},paperInfor_L[whatClassUsed]]
            if paperId in allReviewerPaper_S:
                continue
            if paperInfor_L[5]<manuscriptYear_L[0] or paperInfor_L[5]>manuscriptYear_L[1]:
                continue
            if not paperInfor_L[0] or not paperInfor_L[4] or len(paperInfor_L[0])==0 or len(paperInfor_L[4])==0:
                continue
            text = paperInfor_L[0] +' '+ paperInfor_L[4]
            if manuscriptWordNum_L[0] > len(text.split()) or len(text.split()) > manuscriptWordNum_L[1]:
                continue
            for c in paperInfor_L[whatClassUsed]:
                if c not in class_reviewerName_D:
                    continue
                reviewer_class_L[0].update(class_reviewerName_D[c])
            if len(reviewer_class_L[0])<manuscriptReviewerNum_L[0] or len(reviewer_class_L[0])>manuscriptReviewerNum_L[1]:
                continue
            manuscriptId_reviewer_class_D[paperId] = reviewer_class_L
            if len(manuscriptId_reviewer_class_D)>=manuscriptNum:
                break
        assert manuscriptId_reviewer_class_D,'无满足条件的稿件!'
        reviewerPaperYearMaxMinAve_L = [0,1000000,0]
        reviewerPaperWordNumMaxMinAve_L = [0,1000000,0]
        reviewerPaperNumMaxMinAve_L = [0,1000000,0]
        reviewerClassNumMaxMinAve_L = [0,1000000,0]
        reviewerStrangeClassNumMaxMinAve_L = [0,1000000,0]
        reviewerAllPaperNum = 0
        allReviewerClass_S = set()
        manuscriptYearMaxMinAve_L = [0,1000000,0]
        manuscriptWordMaxMinAve_L = [0,1000000,0]
        manuscriptReviewerNumMaxMinAve_L = [0,1000000,0]
        manuscriptClassNumMaxMinAve_L = [0,1000000,0]
        allManuscriptClass_S = set()
        for name,inf in tqdm(reviewerName_paperId_class_D.items(),mName+'-统计审稿人信息'):
            reviewerAllPaperNum += len(inf[0])
            for paperId in inf[0]:
                p = self._id_title_author_c1_c2_abstract_year_D[paperId]
                text = p[0] + ' ' + p[4]
                year = p[5]
                wordNum = len(text.split())
                if reviewerPaperYearMaxMinAve_L[0]<year:
                    reviewerPaperYearMaxMinAve_L[0] = year
                if reviewerPaperYearMaxMinAve_L[1]>year:
                    reviewerPaperYearMaxMinAve_L[1] = year
                reviewerPaperYearMaxMinAve_L[2]+=year
                if reviewerPaperWordNumMaxMinAve_L[0]<wordNum:
                    reviewerPaperWordNumMaxMinAve_L[0] = wordNum
                if reviewerPaperWordNumMaxMinAve_L[1]>wordNum:
                    reviewerPaperWordNumMaxMinAve_L[1] = wordNum
                reviewerPaperWordNumMaxMinAve_L[2]+=wordNum
            if reviewerPaperNumMaxMinAve_L[0]<len(inf[0]):
                reviewerPaperNumMaxMinAve_L[0] = len(inf[0])
            if reviewerPaperNumMaxMinAve_L[1]>len(inf[0]):
                reviewerPaperNumMaxMinAve_L[1] = len(inf[0])
            reviewerPaperNumMaxMinAve_L[2]+=len(inf[0])
            if reviewerClassNumMaxMinAve_L[0]<len(inf[1]):
                reviewerClassNumMaxMinAve_L[0] = len(inf[1])
            if reviewerClassNumMaxMinAve_L[1]>len(inf[1]):
                reviewerClassNumMaxMinAve_L[1] = len(inf[1])
            reviewerClassNumMaxMinAve_L[2]+=len(inf[1])
            num = 0
            for c,n in  inf[1].items():
                if n>=howClassNumForQual:
                    num+=1
            if reviewerStrangeClassNumMaxMinAve_L[0]<num:
                reviewerStrangeClassNumMaxMinAve_L[0] = num
            if reviewerStrangeClassNumMaxMinAve_L[1]>num:
                reviewerStrangeClassNumMaxMinAve_L[1] = num
            reviewerStrangeClassNumMaxMinAve_L[2]+=num
            allReviewerClass_S |= set(inf[1])
        reviewerPaperYearMaxMinAve_L[2]/=reviewerAllPaperNum
        reviewerPaperWordNumMaxMinAve_L[2]/=reviewerAllPaperNum
        reviewerPaperNumMaxMinAve_L[2]/=len(reviewerName_paperId_class_D)
        reviewerClassNumMaxMinAve_L[2]/=len(reviewerName_paperId_class_D)
        reviewerStrangeClassNumMaxMinAve_L[2]/=len(reviewerName_paperId_class_D)
        for id,inf in tqdm(manuscriptId_reviewer_class_D.items(),mName+'-统计稿件信息'):
            p = self._id_title_author_c1_c2_abstract_year_D[id]
            text = p[0] + ' ' + p[4]
            year = p[5]
            wordNum = len(text.split())
            if manuscriptYearMaxMinAve_L[0]<year:
                manuscriptYearMaxMinAve_L[0] = year
            if manuscriptYearMaxMinAve_L[1]>year:
                manuscriptYearMaxMinAve_L[1] = year
            manuscriptYearMaxMinAve_L[2]+=year
            if manuscriptWordMaxMinAve_L[0]<wordNum:
                manuscriptWordMaxMinAve_L[0] = wordNum
            if manuscriptWordMaxMinAve_L[1]>wordNum:
                manuscriptWordMaxMinAve_L[1] = wordNum
            manuscriptWordMaxMinAve_L[2]+=wordNum
            if manuscriptReviewerNumMaxMinAve_L[0]<len(inf[0]):
                manuscriptReviewerNumMaxMinAve_L[0] = len(inf[0])
            if manuscriptReviewerNumMaxMinAve_L[1]>len(inf[0]):
                manuscriptReviewerNumMaxMinAve_L[1] = len(inf[0])
            manuscriptReviewerNumMaxMinAve_L[2]+=len(inf[0])
            if manuscriptClassNumMaxMinAve_L[0]<len(inf[1]):
                manuscriptClassNumMaxMinAve_L[0] = len(inf[1])
            if manuscriptClassNumMaxMinAve_L[1]>len(inf[1]):
                manuscriptClassNumMaxMinAve_L[1] = len(inf[1])
            manuscriptClassNumMaxMinAve_L[2]+=len(inf[1])
            allManuscriptClass_S |= set(inf[1])
        manuscriptYearMaxMinAve_L[2]/=len(manuscriptId_reviewer_class_D)
        manuscriptWordMaxMinAve_L[2]/=len(manuscriptId_reviewer_class_D)
        manuscriptReviewerNumMaxMinAve_L[2]/=len(manuscriptId_reviewer_class_D)
        manuscriptClassNumMaxMinAve_L[2]/=len(manuscriptId_reviewer_class_D)
        print('审稿人论文MaxMinAve年份:%s, 审稿人论文MaxMinAve词数:%s, 审稿人论文MaxMinAve数量:%s, 审稿人分类MaxMinAve数量:%s, 具有分类资格的分类MaxMinAve数量:%s'
              '审稿人论文总数:%d, 审稿人总数:%d, 审稿人分类种数:%d'%
              (str(reviewerPaperYearMaxMinAve_L),str(reviewerPaperWordNumMaxMinAve_L),str(reviewerPaperNumMaxMinAve_L),str(reviewerClassNumMaxMinAve_L),str(reviewerStrangeClassNumMaxMinAve_L),
               reviewerAllPaperNum,len(reviewerName_paperId_class_D),len(allReviewerClass_S)))
        print('稿件MaxMinAve年份:%s, 稿件MaxMinAve词数:%s, 稿件审稿人MaxMinAve数量:%s, 稿件分类MaxMinAve数量:%s, 稿件总数:%d, 稿件分类种数:%d'%
              (str(manuscriptYearMaxMinAve_L),str(manuscriptWordMaxMinAve_L),str(manuscriptReviewerNumMaxMinAve_L),str(manuscriptClassNumMaxMinAve_L),
               len(manuscriptId_reviewer_class_D),len(allManuscriptClass_S)))
        self._reviewerName_paperId_class_D = reviewerName_paperId_class_D
        self._manuscriptId_reviewer_class_D = manuscriptId_reviewer_class_D

    def saveReviewerFolder(self,path='a+作者编号_论文内容/',deletePathFile=False):
        assert self._reviewerName_paperId_class_D,'没有审稿人信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path),mName+'-删除原文件夹内容'):
                    os.remove(path + '/' + i)
        for name, inf in tqdm(self._reviewerName_paperId_class_D.items(), mName + '-保存到审稿人文件夹'):
            with open(path+'/'+name+'.txt','w',encoding='utf-8') as w:
                for paperId in inf[0]:
                    w.write('{fenge}\n')
                    p = self._id_title_author_c1_c2_abstract_year_D[paperId]
                    text = p[0] + '\t' + p[4]
                    w.write(paperId+'\t'+text+'\n')

    def saveManuscriptFolder(self,path='a+文档集/',labelName='文档-标准作者排名.txt',segTitleAbstract=' ',deletePathFile=False):
        assert self._manuscriptId_reviewer_class_D,'没有稿件信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path),mName+'-删除原文件夹内容'):
                    os.remove(path + '/' + i)
        for id, inf in tqdm(self._manuscriptId_reviewer_class_D.items(), mName + '-保存到稿件文件夹'):
            with open(path+'/'+id+'.txt','w',encoding='utf-8') as w:
                p = self._id_title_author_c1_c2_abstract_year_D[id]
                text = p[0] + segTitleAbstract + p[4]
                w.write(text)
        with open(path+'/'+labelName,'w',encoding='utf-8') as w:
            for id, inf in self._manuscriptId_reviewer_class_D.items():
                w.write(id+'.txt')
                for reviewerName in inf[0].keys():
                    w.write('\t'+reviewerName)
                w.write('\n')

    def _creatAuthorInfor(self,id_title_author_c1_c2_abstract_year_L,savePath = None):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        authorName_paperId_c1_c2_D = {}
        allAuthor_S = set()
        allC1_S = set()
        allC2_S = set()

        for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(id_title_author_c1_c2_abstract_year_L,mName + '-建立作者信息字典'):
            for author in author_L:
                allAuthor_S.add(author)
                if author not in authorName_paperId_c1_c2_D:
                    authorName_paperId_c1_c2_D[author] = [{},{},{}]
                authorInfor_L = authorName_paperId_c1_c2_D[author]
                authorInfor_L[0][id] = None
                for i in c1_L:
                    if i in authorInfor_L[1]:
                        authorInfor_L[1][i] += 1
                    else:
                        authorInfor_L[1][i] = 1
                    allC1_S.add(i)
                for i in c2_L:
                    if i in authorInfor_L[2]:
                        authorInfor_L[2][i] += 1
                    else:
                        authorInfor_L[2][i] = 1
                    allC2_S.add(i)
        if savePath:
            with open(savePath,'wb') as w:
                b = pickle.dumps(authorName_paperId_c1_c2_D)
                w.write(b)
        print('作者数量:%d, subject分类数量:%d, MSC分类数量:%d'%(len(allAuthor_S),len(allC1_S),len(allC2_S)))
        return authorName_paperId_c1_c2_D

class PaperAnalysis:
    def __init__(self):
        self._id_title_author_c1_c2_abstract_year_L = None

    def startAnalysis(self,xmlFolderPath:str,savePath = None):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        allPath_L = self._getAllXmlPath(xmlFolderPath)
        allPaper_L = []
        for path in tqdm(allPath_L,mName+'-读取xml文件'):
            with open(path,'r',encoding='utf-8') as r:
                text = r.read()
                allPaper_L += self._analysisXmlText(text)
        id_title_author_c1_c2_abstract_year_L = []
        noTitleSum = 0
        noAuthorSum = 0
        noC1Sum = 0
        noC2Sum = 0
        noAbstractSum = 0
        noYearSum = 0
        for paper in tqdm(allPaper_L,mName+'-提取论文信息'):
            id = paper[0]
            title = None
            for i in paper[1]:
                title = self._clean(paper[1][0])
                break
            if not title:
                noTitleSum += 1
            author_L = []
            for i in paper[2]:
                if len(re.findall('[a-zA-Z]', i)) == 0:
                    continue
                i=i.lower()
                author_L.append(i)
            if not author_L:
                noAuthorSum += 1
            c1_L = []
            c2_L = []
            for i in paper[3]:
                if len(re.findall('[a-z]',i)) > 0:
                    if len(re.findall('Primary:|Secondary:|primary:|secondary:',i)) == 0:
                        c1_L.append(i.strip())
                    else:
                        for j in re.findall('(?<=[^0-9A-Za-z.])[0-9A-Z.]+?(?=[^0-9A-Za-z.])',i):
                            c2_L.append(j.strip('.'))
                    continue
                for j in re.split('[,;\s]+',i):
                    c2_L.append(j.strip().strip(' .'))
            if not c1_L:
                noC1Sum += 1
            if not c2_L:
                noC2Sum += 1
            abstract = None
            for i in paper[4]:
                if re.findall('\*\*\*\s*Comments:|\*\*\*\s*comments:','***'+i):
                    continue
                abstract = self._clean(i)
                break
            if len(paper[4])==1 and not abstract:
                abstract = paper[4][0]
            if not abstract:
                noAbstractSum += 1
            year = None
            if paper[5]:
                year = int(paper[5][-1].split('-')[0])
            if not year:
                noYearSum += 1
            id_title_author_c1_c2_abstract_year_L.append([id,title,author_L,c1_L,c2_L,abstract,year])
        print('论文总数:%d, 无标题论文数:%d, 无作者论文数:%d, 无subject分类论文数:%d, 无MSC分类论文数:%d, 无摘要论文数:%d, 无年份论文数:%d'%
              (len(allPaper_L),noTitleSum,noAuthorSum,noC1Sum,noC2Sum,noAbstractSum,noYearSum))
        if savePath:
            with open(savePath,'w',encoding='utf-8') as w:
                for i in tqdm(id_title_author_c1_c2_abstract_year_L,'写入论文信息'):
                    w.write(i[0])
                    for j in i[1:]:
                        w.write('\t'+str(j))
                    w.write('\n')
        self._id_title_author_c1_c2_abstract_year_L = id_title_author_c1_c2_abstract_year_L
        return id_title_author_c1_c2_abstract_year_L

    def _clean(self,text:str):
        text = text.strip().lower()
        text = text.replace('\t',' ')
        text = text.replace('\n',' ')
        text = text.replace('\r',' ')
        return text

    def _getAllXmlPath(self,path:str):
        allPath_L = []
        for fileName in os.listdir(path):
            suffix = os.path.splitext(fileName)[1].lower()
            if suffix == '.xml':
                allPath_L.append(path + '/' + fileName)
        return allPath_L

    def _analysisXmlText(self,text:str):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        allPaper_L = []
        for t in text.split('</record>')[:-1]:
            t=t.replace('\r','')
            t=t.replace('\n','')
            try:
                id = re.findall('(?<=<record id="oai:arXiv.org:)[^"]+',t)[0]
                id = id.replace('/',';')
            except:
                print(t)
                raise 1
            title_L = re.findall('(?<=<dc:title>).+?(?=</dc:title>)',t)
            creator_L = re.findall('(?<=<dc:creator>).+?(?=</dc:creator>)',t)
            subject_L = re.findall('(?<=<dc:subject>).+?(?=</dc:subject>)',t)
            description_L = re.findall('(?<=<dc:description>).+?(?=</dc:description>)',t)
            date_L = re.findall('(?<=<dc:date>).+?(?=</dc:date>)',t)
            allPaper_L.append([id,title_L,creator_L,subject_L,description_L,date_L])
        return allPaper_L

    def readSaveFile(self,path):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        id_title_author_c1_c2_abstract_year_L = []
        with open(path,'r',encoding='utf-8') as r:
            for line in tqdm(r,mName+'-读取数据文件'):
                line = line.strip().split('\t')
                if len(line)<3:
                    continue

                id = line[0]
                title = line[1]
                author_L = eval(line[2])
                c1_L = eval(line[3])
                c2_L = eval(line[4])
                abstract = line[5]
                year = line[6]

                if title == 'None': title = None
                if abstract == 'None': abstract = None
                if year == 'None':
                    year = None
                else:
                    year = int(year)

                id_title_author_c1_c2_abstract_year_L.append([id, title, author_L, c1_L, c2_L, abstract, year])
        self._id_title_author_c1_c2_abstract_year_L = id_title_author_c1_c2_abstract_year_L
        return id_title_author_c1_c2_abstract_year_L

    def writePaperText(self,path):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_title_author_c1_c2_abstract_year_L,'没有论文信息!'
        with open(path,'w',encoding='utf-8') as w:
            for id,title,author_L,c1_L,c2_L,abstract,year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出'):
                w.write(id+'\t'+title+'\t'+abstract+'\n')

    def writeCorpus(self,path):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_title_author_c1_c2_abstract_year_L, '没有论文信息!'
        with open(path, 'w', encoding='utf-8') as w:
            for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出第一遍'):
                w.write(title + '\n')
                w.write(abstract + '\n')
            w.write('\n')
            for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出第二遍'):
                w.write(re.sub('[^0-9a-zA-Z]+',' ',title) + '\n')
                w.write(re.sub('[^0-9a-zA-Z]+',' ',abstract) + '\n')

    def write论文_方向_年份_作者_引用论文_被引论文广义表(self,address,whatClassUsed= 'subject'):
        assert self._id_title_author_c1_c2_abstract_year_L, '没有论文信息!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        论文_方向_年份_作者_引用论文_被引论文广义表 = {}
        for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName + '获得广义表'):
            if whatClassUsed == 'subject':
                c = c1_L
            elif whatClassUsed == 'MSC':
                c = c2_L
            else:
                assert False, '分类错误!'
            方向 = {i:None for i in c}
            年份 = {str(year):None} if year else {'0':None}
            作者 = {i:None for i in author_L}
            论文_方向_年份_作者_引用论文_被引论文广义表[id] = [方向,年份,作者,{},{},{},{}]
        二进制流 = pickle.dumps(论文_方向_年份_作者_引用论文_被引论文广义表)
        with open(address, 'wb') as w:
            w.write(二进制流)

def start():
    ap = ''
    PaperAnalysis_obj = PaperAnalysis()
    # id_title_author_c1_c2_abstract_year_L = PaperAnalysis_obj.startAnalysis(r'K:\数据\arxiv',
    #                             r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_编号_标题_作者_分类1_分类2_摘要_日期.txt')
    id_title_author_c1_c2_abstract_year_L = PaperAnalysis_obj.readSaveFile(r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_编号_标题_作者_分类1_分类2_摘要_日期.txt')
    # PaperAnalysis_obj.writePaperText(r'D:\data\code\data\1-RAP\arxiv数据\arxiv论文编号_英文题目摘要.txt')
    # PaperAnalysis_obj.writeCorpus(r'D:\data\code\data\1-RAP\arxiv数据\arxivDoubleCorpus.txt')
    PaperAnalysis_obj.write论文_方向_年份_作者_引用论文_被引论文广义表(address=r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_论文_方向_年份_作者_引用论文_被引论文广义表.pkl',
                                                    whatClassUsed= 'subject')

    # PaperFilter_obj = PaperFilter(id_title_author_c1_c2_abstract_year_L)
    # PaperFilter_obj.filter(reviewerNum=10000,reviewerYear_L=[0,2015],reviewerPaperWordNum_L=[160,300],reviewerPaperNum_L=[50,150],
    #            manuscriptNum=10000,manuscriptYear_L=[2016,3000],manuscriptWordNum_L=[160,300],manuscriptReviewerNum_L=[20,50],
    #            whatClassUsed = 'subject',howClassNumForQual = 20)
    # PaperFilter_obj.saveReviewerFolder(path=r'D:\data\code\data\1-RAP\输出文件目录-14/a+作者编号_论文内容/',deletePathFile=True)
    # PaperFilter_obj.saveManuscriptFolder(path=r'D:\data\code\data\1-RAP\输出文件目录-14/a+文档集/',deletePathFile=True)

start()