"""大模型类"""


import requests
from typing import List, Optional

from pydantic import BaseModel


# TODO(@zyw): 合并两类模型服务接口的参数格式
class Params(BaseModel): 
    """模型服务接口参数"""
    prompt: str = "hello"
    queries: List[str] = []
    history: List[List[str]] = []
    max_length: int = 8192
    top_p: float = 0.7
    temperature: float = 0.97
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample = True
    max_time: float = 60.0


class ChatModel: 
    """对话模型的基类"""

    def __init__(self): 
        raise NotImplementedError


class ChatGLM2(ChatModel): 

    def __init__(
        self, 
        model_name: Optional[str] = "chatglm2-6b", 
        host: str = None, 
        port: int = None, 
        max_length: Optional[int] = 8192, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7, 
        timeout: Optional[float] = 60.0
    ): 
        self.model_name = model_name
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        # self.api: str = "http://{host}:{port}/chat".format(host=self.host, port=str(self.port))
        self.timeout = timeout
    
    def chat(self, content: str = "") -> str: 
        api: str = "http://{host}:{port}/chat".format(host=self.host, port=str(self.port))
        headers = {"Content-Type": "application/json", }
        params = dict(
            prompt=content, 
            history=[], 
            max_length=self.max_length, 
            top_p=self.top_p, 
            temperature=self.temperature, 
            repetition_penalty=1.0, 
            max_time=self.timeout
        )
        resp = requests.post(api, headers=headers, json=params)
        return resp.json().get("response")


class InternLM(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "internlm-chat-20b", 
        host: str = None, 
        port: int = None, 
        max_length: Optional[int] = 4096, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7
    ): 
        self.model_name = model_name
        self.max_length: int = max_length
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        self.api: str = "http://{host}:{port}/v1/chat/completions".format(host=self.host, port=str(self.port))
        # self.prompt_template: str = PROMPT_TEMPLATE
    
    def _call(self, prompt: str = ""): 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
        }
        return requests.post(self.api, json=payload)
    
    def chat(self, content: str = "") -> str: 
        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")


class ChatGLM3(ChatModel): 
    
    def __init__(
        self, 
        model_name: Optional[str] = "chatglm3-6b", 
        host: str = None, 
        port: int = None, 
        temperature: Optional[float] = 0.97, 
        top_p: Optional[float] = 0.7
    ): 
        self.model_name = model_name
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.host = host
        self.port = port
        self.api: str = "http://{host}:{port}/v1/chat/completions".format(host=self.host, port=str(self.port))
        # self.prompt_template: str = PROMPT_TEMPLATE
    
    def _call(self, prompt: str = ""): 
        payload = {
            "model": self.model_name, 
            "messages": [
                {"role": "user", "content": prompt}
            ], 
            "temperature": self.temperature, 
            "top_p": self.top_p, 
            "max_tokens": 8192
        }
        return requests.post(self.api, json=payload)
    
    def chat(self, content: str = "") -> str: 
        resp = self._call(content)
        return resp.json().get("choices")[0].get("message").get("content")


if __name__ == '__main__': 
    llm = InternLM(
        host="172.21.4.23", 
        port=10375, 
        temperature=0.96
    )

    # for i in range(1): 
    #     temp_prompt = """你好，请给我整理一份南京旅游攻略吧"""
    #     print("第{}轮生成".format(i), llm.chat(temp_prompt))

    # llm = ChatGLM2(
    #     host="172.21.4.23", 
    #     port=10373, 
    #     timeout=2.0
    # )

    # prompt = "你好，请帮我编写一份南京旅游攻略"
    # print(llm.chat(prompt))

    group_messages = """
    不要这样说自己|wxid_kimc3ndv29b632拍了拍Tong-说秦晓彤天天开心|/::Z|@晗（新任群主3号 |没我好看|啧啧啧~|不是|<msg><imgmsg|fileid="3057020100044b30490201000204132250c902032fa73b0204a204e878020464c0209a04243|626261346465662d633430362d343434332d383034302d663934333331323138613238020401290a020|01000405" aeskey="e8e8d6be447bad46478c3226" /></msg>|身体|我昨晚心季|你倒是上抖音看看呀……|<msg><imgmsg|fileid="3057020100044b30490201000204a3dcc78702032dcdcd020420f4f32b020464c01fd204243|373037323737312d623337652d346135382d626366392d383336323936303938373335020401290a020|01000405" aeskey="4c196c829c71232880d71263" /></msg>|和大姑在一起6......|生怕错过他们的信息|闭眼一分钟了|我睡不着|我眼珠子都要哭掉了|我想睡|床旁心电图加收|wxid_kimc3ndv29b632拍了拍。我在在啦|享年15|没病史|？|你觉得|主要是我怕我去医院检查出什么毛病|[Doge]|<msg><imgmsg|fileid="3057020100044b30490201000204132250c902032fa73b0204a204e878020464c0210e04243|633233376265352d626661312d343135302d386561362d316462346132343630383133020401290a020|01000405" aeskey="0563f95c76ff6b6d6d802d70" /></msg>|鼓励我是吧|发送位置行为信令|你吃了他|？|有没有人 出生在一个医院|真有焦虑症我跟侯雨贝就同病相怜了|@有对象的ggbond（新任群主2号） |@Tong- @晗（新任群主3号 你俩觉得有必要吗|/::)|我抖音名字……|/::Z|我去医院|咋的？之前耳朵不好使，现在眼睛也不好使啊|马上|[Doge]|你想怎么吃？|假的|@对方正在输入... 笑什么笑。你也几百年没上抖音了|咳咳|你俩|我是这个傻逼医院的|/::)|家人们我好像被人骗了咋办|哭出来好多血丝|/:weak|wxid_wi1qaor4lczq22拍了拍晗的钱包发现是扁的|红烧，糖醋|@有对象的ggbond（新任群主2号） 咋滴对象快俩月不在把自己逼焦虑了|你这属于黑白颠倒了|侯雨贝回来就好了[Doge]|帮死猪烧纸了|把我踹出去，我奖励520个刘忠发给你|我他妈有你抖音吗|我知道我快手为什么限流了|/:v|/:v|好爱|我爸妈|有没有在这个医院的|[发呆]|我命令你|我今晚不睡|第三|[Doge]|又是个崩溃的夜晚|梁猫菡|我也|《英年早逝》|@黑化小公主 |你是熬夜熬的|好爱|刚生下来活不了的|@纯爱战士（新任群主 火速互关|6。主打一个无理取闹是吧|哥醒了|wxid_9spjrotlsy1f12拍了拍GG bond你个显眼包拍我干嘛|又疯1个|没有|<msg><voicemsg endflag="2" length="5530" voicelength="3160" clientmsgid=""|fromusername="wxid_kimc3ndv29b632" downcount="0" cancelflag="0" voiceformat="4"|forwardflag="0" bufid="0" /></msg>|6|两次手抖个不停|我没有你造谣|@纯爱战士（新任群主 我是保安|不知道现在还招不招|自然死亡是真怕|@有对象的ggbond（新任群主2号） 死猪你焦虑？|我没事了|这是哪个傻逼|乐园？|@顾颜。 |@晚安小番茄 你看看她抖音名|我爸在|要死我也不能先死|<msg><voicemsg endflag="2" length="3107" voicelength="1920" clientmsgid=""|fromusername="wxid_kimc3ndv29b632" downcount="0" cancelflag="0" voiceformat="4"|forwardflag="0" bufid="0" /></msg>|我天天在抖音嗷嗷艾特你们几个|我用小号给大号连赞来着|wxid_kimc3ndv29b632拍了拍苏.|我怕死罢了|美好的一天从睡觉开始|/:v|如果可以我想和她z|我奶奶|关键是她老公这关怎么办|我现在睡|@Tong- 6|我怕死|甲功三项|你需要检查 血常规五项|儿童医院|[Duh]|干什么|<msg><videomsg|fileid="3057020100044b30490201000204a3dcc78702032dcdcd02041ef4f32b020464bc0d0004243|363735393137612d616436392d343938302d386338612d3236623563396433393032380204011800040|01000405" aeskey="c1efb7fe3281d97451949091" /></msg>|wxid_9spjrotlsy1f12拍了拍GG bond你个显眼包拍我干嘛|@Tong- 应该有|鄙视你们|<msg><voicemsg endflag="2" length="2691" voicelength="1580" clientmsgid=""|fromusername="wxid_kimc3ndv29b632" downcount="0" cancelflag="0" voiceformat="4"|forwardflag="0" bufid="0" /></msg>|不要伤感|@Tong- 有|@黑化小公主 你怎么又不说话了|怎么吃|你说啥|6|跟我一样|我想嘟嘟了|连赞官方会以为是营销然后限流|你知道吧|伤感了[Broken][Broken][Broken][Broken][Broken]|把我踹出去，我奖励520个红星星给你|/::)/::)|不听|我给你俩都拉黑|@晚安小番茄 你人呢|啊等我|给我个解决办法|晚上睡|？|在黑化给你踹出去|数学4分，看你们都没我牛逼|好喜欢[Doge]|没啥事|打暑假工|我好烦|你俩清高|啥|/::<|五香|/:weak|@有对象的ggbond（新任群主2号） 请讲下你的病史|死猪赶紧睡觉/::)|啧啧啧|即可知道|wxid_kimc3ndv29b632拍了拍晗的钱包发现是扁的|胸部CT|我已经准备好纸钱|群里玩蛋仔的报一下名字呜呜 我想跟你们一起玩|<msg><videomsg|fileid="3057020100044b30490201000204132250c902032fa73b0204a604e878020464c0115004243|343637373736652d623738362d346431352d613562312d6466353839636234376331360204012800040|01000405" aeskey="973de7f308596eaf1619edc9" /></msg>|@Tong- 你吃我嘟腿？|睡不着不硬睡|@黑化小公主 你快滚|我怕我明早醒不来|<msg><voicemsg endflag="2" length="3029" voicelength="1740" clientmsgid=""|fromusername="wxid_kimc3ndv29b632" downcount="0" cancelflag="0" voiceformat="4"|forwardflag="0" bufid="0" /></msg>|再说了|不认识上来就这么说|/:heart/:heart/:heart|又暴露一次~|@Tong- 你趁狗之危|为什么不问我|TM谁不是因为这个在强撑着|没个1000块|就是说|3点了|又瞒着我|？|/:bye/:bye/:bye|虽然说我抓人不怕死|又干啥了|@晗（新任群主3号 @纯爱战士（新任群主 陪我玩猛鬼宿舍|我抓小偷|我没有|哈哈哈哈哈哈哈哈|愤怒的一天从睡觉被叫醒开始|给我炖了我要吃|我现在就是啦|@有对象的ggbond（新任群主2号） |离谱|侯雨贝|发送位置行为信令|就是这件事|早上坏~|你没疯装病啊| 这种问题对象回来了就好了|发送位置行为信令|也亖在那个医院的|你快滚@有对象的ggbond（新任群主2号） |[撇嘴]|明天白天|玩不起是吧|放下杂念|@小苏. 我这种是不是焦虑症啊|我是不是得做个精神类|/::B|劭总锁骨|/::B|第二 我可能有焦虑症|/::)|把手机丢了|死猪，你不是说要去医院做那个焦虑症吗|啊|选择性失明是吧|在一起|但是又睡不着|打个120|立刻睡觉/::~|爆炒，香辣|两个群那么像哪个正常人分辨得了|现在睡|花费2000|心慌|会不会猝死|/::@|嘤/:break|最近没上抖音|我现在就是一直忙着回复客户的消息|孤立|D2凝聚体|<msg><voicemsg endflag="2" length="2628" voicelength="1760" clientmsgid=""|fromusername="wxid_kimc3ndv29b632" downcount="0" cancelflag="0" voiceformat="4"|forwardflag="0" bufid="0" /></msg>|我为您开检查/:rose|我睡不着的原因|没看出来吗我是在厕所拍的|见过表白的，没见过跟女表白的|真棒~|<msg><imgmsg|fileid="3057020100044b30490201000204749b069c02032df0f1020494248a96020464c022d304246|313235323537382d313363612d343839362d613232382d653331376665373661643332020401250a020|01000405" aeskey="7118eb9c90e9388ef2529fd5" /></msg>|我每个星期都要去医院|生吃|/::)|不看|伤感|发送位置行为信令|生物钟倒过来|奥利奥真好吃|明天白天不睡|不让你出医院/:rose/:@)|你都别想走出医院|需不需要120？|微信小游戏|啊？|什么 你居然要吃黄曲奇@晗（新任群主3号 |[Doge]|给你的那个表情包挡了|我要黑化|@纯爱战士（新任群主 |@黑化小公主 |心理上的全面检查啊|我现在睡了|@黑化小公主 |/::)|心脏彩超|24号的石家庄|/::)|烦|wxid_9spjrotlsy1f12拍了拍GG bond你个显眼包拍我干嘛|那样更容易损害生日|吓到|同意|肾功三项|/::||死猪说他有可能要亖了|我睡下来就在想|肝功八项|好好的|[Broken]|睡不着我再来|[Doge]|你直接说 我喜欢你|我在玩|@黑化小公主 |到晚上再睡|第一 昨晚真被吓到了|我睡觉|你说我是不是真该做个身体上|一个月去两次医院|搁哪儿发呢|不艾特我？|我现在离他几百公里|？伤感什么|摆烂|无理取闹是吧|怕|你造谣|吞药的时候|玩呢|对啊|孤立我俩|我不信|zao|你们倒是回我呀|我不听|行|我要黑化|检查出什么毛病|油炸，清蒸|那我跟你说一下吧，我喜欢我大姑|/:break/:break|说什么？|你闭嘴吧|/::B|黑化|如果我是医生|没软用|我在想|你不花个1500|/::)|就一斑秃|被什么|动态心电图|啥也不想睡觉|死猪你莫得事吧|我有点怀疑我是焦虑症|黑化时间到|行|我睡醒了|我要化黑|你到底有没有毛病|我万一死了|不吃药我睡不着|wxid_0ap6866x8oyg22拍了拍GG bond你个显眼包拍我干嘛|6。你是想打它了吧|我这是黑化，不是伤感|你TM越想越有病|是的|化完妆|快想想怎么办|你懂吧|咋办|各位有没有什么赚钱的方法|/::Z|哇女人你喜欢我吗|不做了|咳咳|<msg><imgmsg|fileid="3057020100044b30490201000204a3dcc78702032dcdcd02041ff4f32b020464c021c404243|393639656163612d663434382d343939372d613038322d343163303038633461343939020401290a020|01000405" aeskey="57018172839c9561ce86f8ce" /></msg>|<msg><imgmsg|fileid="3057020100044b30490201000204132250c902032fa73b0204a204e878020464c0207704243|313033623464642d303739392d343338652d623933312d313632336466336632323331020401290a020|01000405" aeskey="8dd1376ce26592006533973e" /></msg>|再黑化给你叉出去|妈卖批|我还没睡着|心电图，脑电图|为什么会有团体治疗这个东西|@纯爱战士（新任群主 你纯爱战士不够还纯爱战神上了是吧|你俩锁死|我这么决定了|6|检查|给自己作吧|/::$"""

    prompt = group_messages + "\n\n\n 请问上述文本中是否涉及到自杀言论？"
    
    # prompt = "你好"
    print(llm.chat(prompt))