import json, glob, argparse
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import kkannotation
from kkannotation.symbolemb import SymbolEmbedding
from kkannotation.coco import CocoManager
from kkannotation.util.com import makedirs


parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=50)
args = parser.parse_args()


list_last_name  = [
    '秋葉', '芝岡', '染川', '高憧', '近久', '黒沢', '中辻', '根木', '柳沢', '石丸', '浅井', '堤', '宮城', '梅田', '三品', 
    '中上', '新鶴田', '山一', '深沢', '小川', '川邉', '大江', '藤川', '西岡', '上條', '山根', '大古場', '萬', '水谷', '比嘉', 
    '伊倉', '生方', '柏野', '岩川', '倉西', '黒石', '青柳', '作野', '重野', '松瀬', '大田', '能仁', '川島', '植木', '野沢', 
    '野間', '船田', '荒金', '湯川', '植田', '乃村', '桐生', '入船', '竹腰', '船越', '滝沢', '平高', '阿部', '福井', '高野', 
    '岩瀬', '武田', '新美', '豊川', '牛田', '兼島', '大原', '河津', '角山', '浦上', '今林', '伴', '今井', '田島', '寺垣', 
    '小芦', '金丸', '和合', '犬童', '栗田', '山崎', '西村', '國分', '米田', '西山', '外崎', '上原', '齋藤', '川北', '平井', 
    '平石', '坂元', '大池', '溝口', '田山', '吉本', '川俣', '澤中', '小坂', '日高', '坂野', '柴田', '石原', '沖村', '納富', 
    '村田', '今泉', '長谷川', '安東', '菅野', '神山', '徳田', '平池', '松元', '石山', '山本', '吉沢', '小山', '宮迫', '萩原', 
    '藤田', '高田', '馬袋', '池上', '秋田', '筧', '酒井', '矢定', '定野', '志気', '早坂', '片桐', '水摩', '田畑', '美濃', 
    '須藤', '西本', '榊原', '松岡', '奈須', '梶山', '花田', '山岡', '谷本', '笠野', '野澤', '眞鍋', '篠原', '大竹', '久保原', 
    '山戸', '吉崎', '内堀', '武原', '本岡', '上田', '米丸', '三谷', '刀根', '古賀', '大串', '伊東', '伏見', '池永', '貫地谷', 
    '中本', '今川', '小柳', '難波', '長溝', '入澤', '谷野', '曾根', '瓜生', '柳', '陶山', '羽尾', '中石', '永田', '本村', 
    '糸數', '金井', '堂原', '川上', '杉山', '前田', '芦村', '岩原', '山内', '古里', '貞本', '大貫', '小菅', '沓名', '幸田', 
    '笠間', '北島', '田村', '武井', '大峯', '高柳', '中里', '川尻', '小泉', '佐々木', '細川', '井上', '新谷', '牧野', '小松原', 
    '平岡', '北郷', '宮武', '神谷', '岸田', '平野', '横山', '金森', '村越', '能見', '喜井', '鳥飼', '森安', '古澤', '江本', 
    '牟田', '内藤', '三瀬', '小倉', '横西', '夏山', '角浜', '久田', '大西', '立石', '黄金井', '小暮', '梶房', '國弘', '樫葉', 
    '宮野', '上船', '矢後', '永瀬', '久保田', '山元', '古藤', '大沢', '坂井田', '中岡', '潮田', '江口', '柘植', '岩井', '定松', 
    '塚本', '興津', '道見', '今垣', '栃原', '香川', '岡谷', '平尾', '宮脇', '鳥居塚', '小形', '大町', '白木', '折下', '宮之原', 
    '八木', '長', '武富', '真田', '若狭', '坂上', '高山', '登', '今坂', '上中', '丸野', '亀山', '井手', '脇山', '里岡', 
    '梶原', '半田', '野相', '土田', '山地', '楢崎', '松堂', '亀井', '麻生', '春園', '上川', '長谷', '生部', '西川', '田代', 
    '松井', '永松', '神ノ口', '抹香', '大山', '磯村', '中山', '池田', '浅和', '井口', '谷口', '浅田', '野崎', '寳田', '元山', 
    '森口', '茂木', '為本', '川名', '海老原', '浜田', '黒木', '田野邊', '大野', '土屋', '豊村', '櫻井', '渋田', '村井', '町田', 
    '崎野', '小嶋', '関口', '森山', '高塚', '福山', '倉持', '森本', '倉田', '久家', '古田', '矢橋', '来田', '堀内', '塚田', 
    '永井', '木幡', '鈴木', '大庭', '熊谷', '仲口', '重木', '伸澤', '正木', '長嶺', '岩下', '岩本', '中川', '宮原', '工藤', 
    '眞田', '反町', '堀越', '河村', '乙津', '大城戸', '西', '田口', '前出', '播口', '鋤柄', '菅', '小杉', '池畑', '本田', 
    '奧村', '松本', '中澤', '川原', '安岐', '宇留田', '赤坂', '下寺', '村外', '尾嶋', '和田', '津留', '上瀧', 'マイケル', '三根', 
    '辺見', '南野', '湯淺', '瀬戸', '久永', '山ノ内', '稗田', '小沢', '金子', '藤井', '小宮', '間嶋', '久富', '千田', '柏木', 
    '金児', '大釜', '阿波', '東', '深山', '丸山', '久間', '阪本', '是澤', '出口', '中間', '藤沢', '竹間', '木谷', '板橋', 
    '倉重', '矢崎', '数原', '平瀬', '江頭', '久保', '友永', '廣瀬', '森作', '石橋', '川田', '志々目', '三木', '岩津', '坂井', 
    '戸塚', '金沢', '貞兼', '沼田', '品川', '大森', '菊原', '大豆生田', '市橋', '木村', '宇佐見', '郷原', '飯田', '沖口', '馬場', 
    '池本', '中元', '西原', '甲斐', '浅野', '諏訪', '板倉', '楮', '益田', '田上', '宮村', '中沢', '五十嵐', '出畑', '土井', 
    '藤堂', '加瀬', '明野', '北村', '杉田', '古結', '峰', '古川', '菅原', '小野寺', '進藤', '佐々野', '四宮', '上平', '嶋田', 
    '三川', '黒明', '鶴田', '荒田', '徳増', '中', '沖', '新開', '木下', '崎', '高濱', '松山', '富山', '真庭', '中世古', 
    '吉島', '石間', '村岡', '荒川', '北中', '上島', '紫垣', '堀江', '安藤', '中井', '長田', '加藤', '汐崎', '平松', '津田', 
    '濱本', '及川', '信濃', '荒井', '手塚', '細谷', '真子', '柾田', '江藤', '桑原', '沢田', '一宮', '丹下', '吉武', '小羽', 
    '深尾', '根岸', '富永', '畠山', '牧', '山下', '浜崎', '奥平', '宮西', '有賀', '芹澤', '広瀬', '伊久間', '堀川', '仲道', 
    '寺本', '飯野', '竹之内', '原', '小野', '相馬', '梅木', '鎌田', '角川', '杉原', '古谷', '児玉', '杉浦', '大嶋', '遠藤', 
    '竹本', '入江', '石井', '重冨', '畑', '佐口', '今村', '鎌野', '金澤', '三村', '長畑', '泥谷', '宮本', '練木', '長尾', 
    '岡崎', '柳澤', '垂水', '入倉', '若女井', '海老澤', '小松', '百武', '野見山', '沖本', '福田', '深井', '前沢', '高松', '服部', 
    '吉岡', '先山', '彦坂', '飛田', '川井', '福原', '出晴', '茶谷', '内海', '高崎', '浦野', '立山', '武藤', '末永', '早川', 
    '根本', '石田', '勝浦', '岩永', '三野', '黒田', '相原', '濱田', '作間', '渡辺', '掛水', '天野', '栗城', '岩木', '角谷', 
    '鈴谷', '馬野', '後川', '深澤', '角', '大谷', '谷川', '國井', '嶌', '河合', '川下', '川野', '品田', '飯塚', '加倉', 
    '丸尾', '野村', '柚木', '新名', '秋元', '篠田', '重富', '三島', '桑島', '柿本', '鶴本', '平子', '塩崎', '向', '大岩', 
    '川添', '宮田', '笠原', '安田', '小城', '深見', '望月', '一色', '亀本', '石岡', '黒柳', '湯浅', '奧田', '下條', '深谷', 
    '益子', '江並', '横川', '広田', '安部', '藤家', '稲垣', '海野', '上園', '園田', '三角', '森川', '本部', '谷添', '田崎', 
    '淺田', '長島', '實森', '星', '畔柳', '栢場', '多田', '岸川', '島村', '稲毛', '花本', '前川', '関', '鍵和田', '竹中', 
    '奥田', '油浦', '國浦', '酒見', '北原', '稲田', '向井田', '黒澤', '中野', '志道', '角井', '柳田', '魚谷', '辻林', '山川', 
    '冨好', '榎本', '市村', '柳生', '門間', '竹田', '沖島', '小田', '籾山', '宮内', '井町', '砂長', '大上', '桜井', '幸本', 
    '滝川', '江原', '梶野', '田内', '白石', '粒崎', '上村', '安井', '粟田', '枝尾', '福嶋', '本井', '八島', '宮沢', '守田', 
    '本多', '三原', '有田', '白水', '赤池', '大須賀', '板谷', '山谷', '北野', '佐川', '妻鳥', '芦澤', '林', '大木', '下出', 
    '石野', '大久保', '繁野谷', '下河', '新地', '新田', '新井', '村本', '水口', '森', '山出', '若杉', '濱野', '田鍋', '出村', 
    '行谷', '藤原', '梅原', '船岡', '竹上', '東本', '大島', '高島', '二橋', '椎名', '渥美', '石黒', '長嶋', '福岡', '伏田', 
    '大川', '菊池', '増沢', '万谷', '高岡', '垣内', '島田', '丹羽', '浮田', '赤岩', '胡本', '石崎', '水野', '島崎', '小池', 
    '野辺', '清水', '長沢', '村松', '船津', '坪井', '内田', '北岡', '片山', '黒瀬', '玄馬', '佐竹', '荻野', '表', '鎌倉', 
    '徳永', '宮崎', '安河内', '櫻本', '高瀬', '中田', '笠置', '橋本', '山路', '浦本', '田嶋', '北川', '村瀬', '榎', '宮井', 
    '筒井', '矢野', '乙藤', '鳥居', '松崎', '山村', '権藤', '松江', '安達', '嶋', '金光', '赤澤', '柳内', '義藤', '勝野', 
    '浦田', '新出', '待鳥', '塩田', '塚脇', '三嶌', '中北', '登玉', '伊藤', '常盤', '真保', '東出', '桐村', '佐野', '吉川', 
    '石塚', '廣町', '大澤', '大橋', '宇野', '入海', '渡部', '津川', '増田', '芹口', '岡野', '臼井', '中島', '栗原', '間庭', 
    '砂川', '西影', '牧山', '守屋', '三輪', '塚崎', '立間', '岸本', '奥山', '大熊', '奥保', '小澤', '樋江井', '荒木', '關戸', 
    '堀口', '中嶋', '松村', '関野', '森清', '森高', '前野', '青木', '竹村', '國崎', '深水', '中西', '松下', '土性', '水長', 
    '藤丸', '森野', '楠原', '平山', '大坪', '横澤', '中越', '花井', '中原', '太田', '打越', '野田', '浜本', '立具', '浅見', 
    '関根', '坂谷', '冨成', '中尾', '木山', '大瀧', '廣光', '宮川', '赤峰', '田中', '土澤', '鰐部', '澤田', '森田', '野中', 
    '寺島', '原田', '香月', '小西', '岡村', '宮嵜', '倉橋', '妹尾', '近江', '福永', '本間', '前本', '柳川', '若林', '下野', 
    '向井', '川村', '新城', '加納', '岡本', '細田', '眞鳥', '山口', '照屋', '澤崎', '三浦', '雨宮', '松原', '須田', '福島', 
    '谷', '奥村', '横田', '東海林', '島川', '橋谷田', '南部', '田川', '切田', '西田', '吉井', '光山', '土井内', '土山', '冨名腰', 
    '倉尾', '多羅尾', '渕田', '大屋', '庄司', '西澤', '浜野', '坪内', '佐藤', '沖原', '伊達', '蜂須', '大城', '山中', '織田', 
    '三好', '辻', '堀', '柳瀬', '孫崎', '渋谷', '稲葉', '高月', '広次', '宇土', '小島', '戸敷', '楠本', '磯部', '大槻', 
    '古場', '桑村', '吉村', '川崎', '本吉', '尾形', '吉田', '村島', '堀田', '伏島', '別府', '井内', '河上', '青野', '谷津', 
    '米井', '重成', '椎原', '坂東', '藤野', '林田', '小林', '小出', '藤崎', '濱村', '森岡', '秦', '桐本', '島倉', '檀', 
    '下村', '五反田', '西野', '橋爪', '榮田', '大茂', '鹿島', '竹野', '秋山', '松竹', '宗行', '二宮', '西沢', '山室', '大河内', 
    '富田', '田原', '柳橋', '金城', '渡邉', '田添', '田頭', '志村', '堀之内', '瀬尾', '横家', '梅崎', '岩田', '計盛', '吉永', 
    '藤本', '中村', '川本', '金谷', '木塚', '三苫', '富樫', '今野', '荘林', '戸田', '濱野谷', '坪口', '石倉', '小寺', '徳平', 
    '石川', '木田', '仲', '喜多', '毒島', '水上', '永嶋', '西坂', '梅内', '宇田川', '大井', '岡瀬', '森林', '池浦', '内野', 
    '田邉', '桂林', '喜川', '杉本', '飯山', '原口', '中谷', '小原', '横畠', '越智', '森定', '山田', '深川', '大賀', '大場', 
    '水原', '黒野', '向後', '吉原', '間野', '新藤', '伯母', '奥野', '濱崎', '杢野', '福本', '岩科', '三ッ井', '勝元', '尾上', 
    '野田部', '峰重', '北山', '斉藤', '竹井', '平田', '中林', '齋宮', '葛原', '内山', '新良', '仁科', '永滝', '泉', '笠', 
    '森年', '金平', '渡', '高木', '山来', '尾崎', '阿波連', '河原', '藤山', '池', '森下', '日笠', '加木', '森久保', '勝又', 
    '草場', '出本', '野添', '西島', '篠崎', '宮地', '落合', '横井', '津久井', '篠木', '滝澤', '岸蔭', '貞廣', '烏野', '豊田', 
    '屋良', '小巻', '藤生', '武重', '鍛治', '芝田', '清埜', '竹下', '緒方', '村上', '佐久間', '山野', '川合', '長野', '三小田', 
    '栄', '大平', '坂', '向所', '橋口', '小黒', '高沖', '今出', '小玉', '東郷', '妻夫木', '宮下', '齊藤', '西舘', '西尾', 
    '浜地', '薮内', '神田', '熊本', '赤羽', '野末', '高倉', '白井', '坂田', '松浦', '大内', '蒲原', '下田', '佐伯', '高石', 
    '松野', '明石', '矢島', '鵜飼', '水本', '江夏', '喜多須', '寺田', '山岸', '大神', '寺嶋', '末武', '飯島', '江崎', '南', 
    '道上', '藤森', '盛本', '荒牧', '栗山', '野長瀬', '神里', '星野', '宇恵', '岡', '大塚', '江野澤', '畑田', '小神野', '谷村', 
    '生田', '吉野', '坂本', '足立', '岩橋', '長岡', '神開', '松島', '廣中', '仲谷', '片岡', '斎藤', '成貞', '常住', '原村', 
    '金山', '坂口', '市川', '森脇', '窪田', '藤村', '丸岡', '近藤', '堀本', '平本', '小笠原', '大石', '三井所', '渡邊', '竹内', 
    '赤井', '石渡', '大崎', '君島', '井芹', '須江', '東口', '河内', '高橋', '関谷', '金田', '二瓶', '小畑', '西橋', '刑部', 
    '藤岡', '相沢', '栩本', '前原', '川口', '川端', '谷田', '米山', '千葉', '三松', '岩谷', '後明', '稲崎', '瀧川', '河野', 
    '永野', '後藤', '一柳', '中渡', '片橋', '成田', '猿田', '幸野', '館野', '岩崎', '平川', '澤', '羽野', '仲野', '平賀', 
    '村山', '三宅', '高辻', '樋田', '室田', '黒崎', '淺香', '菊地', '上之', '黒井', '井本', '岩口', '松尾', '角田', '杉江', 
    '浜先', '大村', '笠井', '若山', '占部', '岸', '大月', '小谷', '折上', '雑賀', '井川', '倉谷', '茅原', '青山', '井坂', 
    '瀬川', '木内', '本橋', '小森', '十河', '古野', '杉村', '早馬', '糸平', '岡田', '平見', '松永', '蜷川', '樋口', '三上', 
    '菅沼', '稲生', '一瀬', '木場', '上野', '久次米', '岡部', '松田', '日野', '田辺', '野口', '玉生', '高井', '武岡', '瀧本', 
    '網谷', '都築', '白神', '八十岡', '福来', '田路', '富澤', '圓井', '黒川', '森永', '茂垣', '塚原', '岡山', '山形', '冨田', '牧原', '河相', '弘中'
]

list_first_name = [
    '智裕', '吉鎬', '匡紀', '秀祈', '徳夫', '康明', '佳子', '大河', '泰治', '麻衣', '智至', '明人', '海輝', '新', '明菜', 
    '吉三', '素也', '正治', '貫太', '空依', '昌敏', 'りな', '夢楠', '理', '将浩', '加菜', '由珠', '裕樹', '美亜', '英史', 
    '暢之', '主樹', '範政', '真帆', '優美', '秀彦', '智一', '将人', '信哉', '史之', '塁', '隆弘', '友史', '義孝', '勇志', 
    '京志郎', '慧', '征夫', '正輝', '陸斗', '史吉', '隼百', '康男', '弘之', '昌司', '真弓', '晋六', '雅夫', '信佑', '種生', 
    '匡博', '優子', '和輝', '美智子', '重則', '恭祐', 'なつみ', '政志', '良輔', '耕輔', '敏樹', '直輝', '恵介', '玲奈', '秀夫', 
    '径冶', '里穂', '夕貴奈', '育美', '栄蔵', '實', '湧', '四季', '峰由季', '幸輝', '八重子', '誠司', '亮次', '清貴', '淑子', 
    '正和', '丈', '敏明', '広幸', '勝也', '庸平', '太己', '友晴', '允珠', '好宏', '利継', '宏紀', '雷太', '章哉', '昌樹', 
    '直喜', '知可', '真史', '佳幸', '保雄', '久男', '綾子', '裕隆', '博文', '文耶', '高雅', '一也', '泰平', '信次', 'ヒデキ', 
    '奈帆子', '海', '峻介', '陽祐', '貴洋', '知樹', '覚', '麦', '雄治', '良昌', '信行', '龍一', '三弘', '哲男', '明美', 
    '智和', '定美', '吉朗', '茂', '茂高', '幸貴', '俊樹', '昭廣', '政信', 'かなえ', '亮輔', '晶恵', '智博', '悠稀', '光男', 
    '久剛', '成吉', '太一', '勇', '友和', '文隆', '基成', '忍三', '伸吉', '一男', 'まり', '善生', '大佑', '吉範', '泰弘', 
    '靖亜', '宏之', '修市', '澄雄', '孝二', '利寿', '久志', '晃嗣', '昌史', '優作', '博雄', '浩喜', '成聡', '渉', '建策', 
    '晃美', '義人', '敏男', '沙友希', '典子', '真梨菜', '友次', '康代', '政昭', '要', '葉月', '佳奈', '克行', '保', '正太郎', 
    '良昭', '邦好', '祐次', '洋希', '紘章', '早紀', '満郎', '知佳', '加奈子', '大作', '和弘', '邦功', '雅一', '侃', '哉', 
    '峻', '智啓', '裕己', '智亮', '泰章', '謙治', '正喜', '直志', '章平', '亮治', '剛大', '友貴', '羊一', '尚久', '勇貴', 
    '茂一', '英人', '匠', '省吾', '豪', '康雅', 'つぐみ', '勝匡', '龍', '晃生', '雅和', '智志', '拡郎', '虎親', '哲', 
    '耀', '利彦', 'アルム', '貴司', '忠志', '良彦', '和幸', '雄朗', '晟恒', '佐季', 'るみ子', '究', '亨', '郁弥', '佑典', 
    '梓', '利仁', '涼斗', '小葉音', '勇哉', '信義', '文典', '潤樹', '愛子', '仁紀', '早菜', '睦', '雅昭', '修平', '彪也', 
    '德治', '旭', '雅美', '奈緒子', '健造', '映二', '重哉', '秀文', '光太郎', '爲秀', '栄祐', '雄貴', '紗希', '賢四郎', '三男', 
    '将亨', '香奈子', '正也', '賢也', '保徳', '小槙', '竜吾', '公洋', '智宏', '勇輝', '奈穂', '愼祐', '昌江', '友', '承則', 
    '浩輔', '洪弥', '啓史朗', '友翔', '正之', '波美音', '新太郎', '凜', '祐季', '明', '夢斗', '春三', '美眞', '幸夫', '孝斗', 
    '文博', '和佐', '京香', '亮平', '準', 'なづき', '雅司', '弘基', '幸司', '哲平', '沙樹', '聖司', '周', '龍也', '温志', 
    '俊郎', '幸信', '絹代', '邦彦', '基樹', '汰一', '兼士', '和信', '玲子', '浩一郎', '大成', '大助', '敦史', '莉里佳', '大我', 
    '正知', '秀明', '真菜', '英勝', '達希', '峻輔', '晴哉', '幸典', '秀雄', '裕', '英夫', '和伸', '貴久', '時政', '雄平', 
    '麻奈美', '洋一郎', '雄太', '真里子', '未奈実', '和樹', '晶', '康臣', '百加', '淳二', '文雄', '拓己', '勲', '英男', '盛也', 
    '晃太', 'りか', '謙次', '敬碁', '龍太郎', '好弘', '祐介', '豊弘', '昇', '強太郎', '裕二', '富士男', '英樹', '友也', '康宏', 
    '数広', '正嗣', '祐一', '佑', '忠寛', '真司', '節也', '弥生', '守', '洋史', '伸忠', '浩一', '忠治', '郁', '弘文', 
    '政吾', '克毅', '友弥', '春華', '幸弘', '昭広', '裕介', '達矢', '魁生', '北斗', '弘幸', '靖', '辰弥', '登', '優太', 
    '暢孝', '友樹', '順', '賢太', '剛', '恵祐', '光広', '克也', '佐英子', '翔平', '明日美', '功', '祐也', '広樹', '元輝', 
    '睦広', '葛', '貞治', '忠良', '君夫', '祐樹', '祥', '浩子', '百音', '梨花', '元泰', '弘美', '隆幸', '佑来', '仁志', 
    '裕平', '耕資', '雅文', '田代', '圭', '成介', '尚人', '翔斗', '明志', '康志', '崇典', '恵助', '正紀', '昭彦', '敏弘', 
    '宏次', '悠司', '晴之', '昂介', '太一郎', '諭', '恭嗣', '美鹿子', '千登志', '慎治', '洋次朗', '通泰', '秀幸', '孝平', '峰晴', 
    '央士', '直己', '泰就', '崇文', '圭子', 'ひかる', '敏幸', '宏美', '等', '宥鎭', '哲治', '政雄', '勝美', '弘喜', '大一', 
    '幸宏', '広大', '美紀', '祐丞', '京平', '悠衣', '博人', '真彦', '拳人', '知江美', '兼司', '恵理香', '健人', '輝彦', '誉士', 
    '一真', '祐明', '志津江', '哲実', '誠治', '太郎', '健介', '荷英', '陽平', '豊隆', '実', '宗孝', '菜摘', '大騎', '尚汰', 
    '誠二', '駿介', '麻衣子', '正弥', '慎', '孝之', '和仁', '竜也', '格', '成樹', '周樹', '安奈', '賢次', '重幸', '寿男', 
    '裕太', '一', '芳男', '一行', '宏樹', 'ちさと', '広和', '昌也', '愛', '友尚', '友恵', '昭光', '教年', '崚雅', '俊一', 
    '弘雅', '竜太朗', '和則', '昌平', '昭三', '晃大', '暢彦', '雄祐', '正浩', '雄吉', '輝年', '良也', '正孝', '博司', '勇輔', 
    '成希', '由美子', '茂登子', '悠祐', '太樹', '忠則', '夏樹', '功祐', '浩平', '勇人', '照美', '祥平', '一恵', 'かほ美', '啓行', 
    '訓', '風太', '暢嵩', '祥子', '真', '丈史', '清一', '建太郎', '徳克', '平三', '慶尚', '重敏', '怜実', '詳子', '祐臣', 
    '友紀恵', '俊介', '駿', '咲渡子', '聖也', '慶一', '行延', '美宏', '辰雄', '憲明', '了', '伸正', '衣織', '正人', '正己', 
    '有佑', '稔宏', '晃', '亜紀', '恵里奈', '晃弘', 'はるか', '諒祐', '保宏', '花夢', '泰史', '健一', '元志', '妃佐', 'イリナ', 
    '雅明', '直博', '真之介', '満弘', '佑樹', '善庸', '直行', '享', '由香', '明里', '将太', '伸也', '陸', '優公', 'エミ', 
    '秀則', '聖賢', '太', '孝志', '城啓', '芽唯', '愛未', '康二', '辰彦', '和孝', '有紀', '峻二', '新吾', '督生', '祐美子', 
    '和士', '利行', '和義', '博', '寛之', '直弘', '敬一', '正剛', '幸香', '輝季', '宏奈', '源', '貞明', '貴美子', 'ひとみ', 
    '義則', '悟', '興志', '忠幸', '力也', '祥史', '雅子', '浩則', '義生', '慶太', '智幸', '祐太郎', '聡志', '純一郎', 'さおり', 
    '真奈美', '洋介', '圭右', '博倫', '優花', '和久', '栄一郎', '富重', '大夢', '龍星', '尚純', '聖秀', '泰', '優香', '光弘', 
    '俊明', '宗臣', '桂三', '和巳', '里香', '寛章', '芳宏', '菜々', '康太', '浩太郎', '慎二', '和政', '忠司', '竜一', '英悟', 
    '太男', '英則', '照浩', '泰蔵', '謙一郎', '芳紀', '百世', '梨菜', '康司', '夏実', '久子', '勁介', '勝貴', '武司', '通治', 
    '智久', '翔悟', '光平', '倫太朗', '貴裕', 'あやの', '菜央', '政之', '貴博', '嘉嗣', '逸人', '章博', '裕貴', '幹雄', '瑞希', 
    '謙侑', '聖悟', '将史', '啓司', '稔太朗', '万記', '善裕', '伸二', '錬志', '学', '富士雄', '恭史', '啓', '桃佳', '皓介', 
    '千晶', '康友', '啓文', '恵理', '芳夫', '博美', '光宏', '美和子', '珉圭', '博憲', '敬', '純', '拓', '利光', '敏裕', 
    '風葵', '優也', '直子', '知音', '慎一郎', '修一', '詩織', '明日香', '雅洋', '佳岳', '文広', '凪紗', '祐一郎', '美香子', '親王', 
    '春繁', '幸造', '雅佳', '友汰', '孝彰', '康幸', '英治', '英輝', '昌宏', '玄太', '敦宏', '隆太郎', '生奈', '正晴', '隆仁', 
    '翔太', '稔', '錦一', '卓人', '博史', '国光', '照正', '直弥', '瑞生', '峻佑', '省三', '望', '小百合', '真子', '忠宏', 
    '知輝', '英彦', '竜生', '那由夏', '麻由美', '正一', '章太', '将吉', '実沙希', '卓巳', '徳雄', '果里', '知美', '昌男', '美香', 
    '俊光', '直一郎', '竜次', '由里子', '詔子', '泰教', '洋', '佑里', '英司', '奨太', '重次', '亮介', '奈津美', '正俊', '太朗', 
    '勤', '宏子', '青海', '都', '巴恵', '聖二', '遊雅', '拓民', '旦明', '輝紀', '恵美', '恭兵', '晟弥', '留美', '彰人', 
    '雅希', '謙冴', '信治', '高弘', '正幸', '良太', '貢輝', '雄史', '颯太', '靖菜', '康行', '浩士', '健策', '佳蓮', '輝義', 
    '佐緒里', '奈緒', '秀司', '琴音', '智広', '慎一', '公二', '逸子', '靖博', '亜里紗', '和夫', '勝利', '幸太郎', '爾士', '夕貴', 
    '辰治', '航平', '隆二', '慎太郎', '奈未', '道男', '潮', '和裕', '喜一郎', '展弘', '香松', '京也', '明寛', '義弘', '千草', 
    '明男', '暁広', '侑我', '妙子', '正広', '清隆', '泰行', '正則', '俊祐', '右貴', '享子', 'まどか', '勇三', '将隆', '進司', 
    '真実子', '凪沙', '愛実', '雅幸', '年博', '道也', '祥二', '節男', '文香', '葵和子', '正行', '泰和', '美代', '篤光', '孝宏', 
    '将也', '剛規', '武裕', '弘志', '治美', '哲二', '麗加', '幸男', '完知', '真希', '雄太郎', '美由紀', '美翼', '大貴', '鉄兵', 
    '大斗', '真奈', '省一', '晋也', '涼花', '千恵', '宣恵', '相禹', '昌志', '圭介', '躍', '征嗣', '二朗', '三貴子', '三知治', 
    '幸美', '敏司', '貢', '貴志', 'ティエン', '誉史', '孝', '喜久子', '厚成', '秀路', '壽賀夫', '侑征', '菜希', '高行', '幸治', 
    '寿人', '光一', '博士', '輝吉', '花凪', '義隆', '祐二', '展大', '博治', '世里', '秀一', '譲', '海心', '和志', '理夏', 
    '和典', '一人', '莉々', '極', '敬司', '育未', '澄夫', '晃三', '真至', '一吉', '照夫', '喜彦', '光', '朋子', '義紘', 
    '真二', '巧', '広一', '浩次', '治義', '義信', '雄希', '繁雄', '紫乃', '圭史', '浩仁', '文武', '政次', '勝弘', '昭利', 
    '類', '徹', '加央理', '孝紀', '秀治', '孝雄', '純一', '茂幸', '勉', '慎二郎', '圭吾', '亜紗美', '聖仁', '昇一', '誠志', 
    '聖文', '崇', '拓郎', '奈美子', '達也', '英諒', '千鶴', '一久', '崇夫', '美憲', '海人', '栄里佳', '出', '一三六', '英喆', 
    '佑紀', '賢斗', '雅義', '龍之介', '永', '奏美', '豪洋', '雄三', '義', '冨美雄', '飛翔', '雅晟', '光章', '憲幸', '兼礼', 
    '直久', '保昌', '康平', '瑞紀', '賢治', '海渡', '宏', '陸翔', '由貴', '一平', '久夫', 'るり華', '江己', '未華', '拓也', 
    '光基', '力良', '完太', '敏', '圭司', '修二', '昌希', '千佳', '麻起子', '流心', '康孝', '由資', '浩貴', '愛海', '宏太', 
    '良至', '洸希', '憲哉', '幸栄', '輝夫', '聖子', '剛治', '碧生', '陽一', '桐加', '賢', '真治', '孝弘', '真太郎', '友治', 
    '和廣', '岳', '繁美', '茂樹', '智子', '美好', '美里', '文英', '美琴', '健一郎', '由子', '勇樹', '健也', '絵里奈', '貴支', 
    '明治', '千聖', '政浩', '和敏', '航太', '俊弘', '美鈴', '朋美', '道臣', '晋太郎', '亮', '拓朗', '永理', '吉弘', '敬太', 
    '嘉広', '萌華', '達哉', '恵', '励', '斗馬', '武', '完信', '夏海', '周平', '浩司', '美保', '俊法', '乃絵', '賢洋', 
    '茂実', '真輔', '茂光', '日紀太', '豊土', '恵子', '恭司', '和也', '馨', '洸太', '征伸', '優吾', '祐輝', '豊年', '博亮', 
    '靖広', '信一郎', '喜智', '法大', '秀行', '誠一', '大輔', '宏志', '泰洋', '政憲', '勝幸', '健太郎', '則雄', '三紀', '彩加', 
    '英一', '隆晟', '大將', '俊秀', '正志', '明義', '淳史', '里枝子', '省二', '千秋', '里彩', 'つかさ', '経夫', '栄二郎', '政弘', 
    '現植', '隆史', '敬介', '翔大', '尚也', '勇生', '哲秀', '彩馨', '律貴', '真弥', '孝四朗', '輝雄', '憲二', '佳江', '麻未', 
    '勝', '勝広', '尚之', '裕也', '亮太', '啓輔', '元', '元基', '奈央', '岳人', '美穂子', '将光', '千広', '英明', '弘', 
    '宗平', '香里', '一規', '顕心', '康晴', '廣志', '宗弘', '義明', '真紀', '由行', '忠義', '一郎', '明広', '努', '奏恵', 
    '栄章', '寧々', '康子', '里奈子', '将季', '竜太', '樹蘭', '聖美', '隆信', '孝男', 'ゆかり', '凌雅', '孝仁', '俊成', '友理', 
    '年光', '由樹夫', '加一', '晋二', '新次', '克己', '大二', 'さくら', '文夫', '歩', '建二', '雄一', '隆浩', '稔弘', '敏之', 
    '博紀', '友輔', '直司', '一輝', '留男', '夏鈴', '航', '頼房', '誠', '真由子', '麻友', '直也', '晶夫', '行晴', '隆太', 
    '祐貴', '徳行', '誠一郎', '保孝', '敬義', '一聡', '友里乃', '美祐', '一朗', '浩明', '真樹', '英昭', '秀志', '隆臣', '智哉', 
    '啓太', 'ティック', '佳伸', '竜馬', '義晴', '忍', '喜一', 'なぎさ', '蓮', '満則', '知哉', '進', '匡宏', '千里', '千亜希', 
    '秀樹', '達夫', '久也', '昭博', '雅人', '里江', '修次', '武彦', '彰', '賢人', '絢也', '昌成', '卓矢', '靖弘', '正明', 
    '理司', '義和', '哲三', '純奈', '風花', 'コイ', '比呂賀', '真之', '義雄', '愛可', '昭宏', '祥司', '圭大', '一雄', '照雄', 
    '綾', '昌克', '颯仁', '宗司', '勇雄', '美穂', '志音', '昂明', '秀徳', '吉夫', '直人', '宗太郎', '伸吾', '充', '規弘', 
    '圭浩', '博明', '貴之', '聡介', '美岐', '公子', '信一', '由紀英', '和広', '堅', '千依', '貴輝', '篤志', '尚', '拓馬', '波乙', '佑治', '奈菜', 
    '文彦', '信幸', '紗友希', '賢一', '孝典', '恒幸', '晃一', '真喜子', '彩乃', '雄介', '志翔', '元樹', 
    '雅弘', '壮志郎', '知博', '宰成', '良一', '遼太', '良二', '亜理沙', '優', '真一', '望美', '郡', '浩哉', '章', '星子', 
    '真美', '繁明', '大輝', '政人', '博之', '樹里', '愛美', '庄吾', '順一', '英志', '寛久', '進一', '孝輔', '訓靖', '範光', 
    '勝敏', '孝義', '拓児', '舞有子', 'あゆみ', '薫', '小六', '豊', '正彦', '治代', '周三', '治哉', '次利', '元浩', '智恵', 
    'HR担当', '千晴', '和美', '将典', '政利', '具巳', '幹太', '健次', '淳一', '智加', '健三', '寿二郎', '魁', '和之', '智巳', 
    '辰己', '晋司', '仁照', '廉', '正弘', '颯', '政勝', '海義也', '実明', '暁', '恒季', '友吾', '四十四', '昂大', '通', 
    '竜弘', '敏夫', '藍', '順子', '賢志', '亜衣花', '克彦', '信夫', '康之', '翔麻', '真歩', '綾菜', '望生', '蒼', '義昭', 
    '竜輔', '佳昭', '章央', '和将', '正義', '有香', '貴也', '弘行', '竜矢', '郁人', '貴愛', 'アハマド', '武日児', '侑', '佳武生', 
    '俊吾', '将之', '昂章', '紀夫', '太二郎', '真吾', '新治', '恵一', '尊春', '龍馬', '昌一', '玲緒', '雄二', '大将', '伸太郎', 
    '竜司', '政哉', '泰二', '隆章', '夏穂', '善行', '宏明', '克洋', '悠紀', '正樹', '朋史', '博昭', '和徳', '辰也', '彰一', 
    '攻二', '利文', '洋成', '侑治', '和郎', '由里', '純也', '数成', '憲一', '崇人', '寛', '英孝', '定雄', '喜実', '貴彦', 
    '康典', '梢', 'アイン', '伸夫', '光紀', '弘司', '正', '久和', '恩夫', '尚悟', '京吾', '由将', '達之', '秒嘉', '杏奈', 
    '健太', '尚治', '明子', '昇司', '年昭', '一豊', '哲史', '賢至', 'クレートン', '康博', '昌弘', '敦', '徳三', '大都', 'ヤエ子', 
    '幸一', '祥伍', '智昭', '宏典', '貴仁', '徹郎', '裕亮', '哲也', '善博', '達徳', '勝正', '卓士', '靖徳', '経太', '幸正', 
    '隆司', '知行', '一哉', 'ユミ', '桂士', '華', '浩孝', '俊洋', '定文', '勝見', '梨湖', '充宏', '善一', '恭裕', '英正', 
    '景士郎', '文明', '陽子', '正哉', '正男', '優里佳', '由加里', '篤哉', '雄次', '由紀', '正昭', '太空海', '康弘', '伸一', '爽佑', 
    '良光', '繁幸', '弥佑紀', '智也', '悠平', '文', '重典', '佑介', '加奈', '一光', 'みゆき', '喬', '稔也', '操拓', '通彦', 
    '喜継', '日向', '泰明', '光希', '優司', '敏行', '和明', '祐希', '宣邦', '逸郎', '敏春', '千夏', '和宏', '勝巳', '勇一', 
    '祐真', '精二', '裕人', '丈一朗', '修', '大志郎', '奈美', '正利', '次郎', '義裕', '秋光', '節明', '利克', '和彦', '益巳', 
    '良夫', '裕之', '紀代子', '裕子', '菜木', '克哉', '通雅', '翔一', '一徳', '智則', '稔子', '順平', '啓介', '一美', '隆雄', 
    '亘', '俊夫', '大陽', '隼之', '純平', '清', '尚哉', '麻里菜', '豊志', '由衣', '由楽', '光雄', '豊治', '和哉', '好一', 
    '紀香', '恭文', '良', '龍健', '良子', '義哲', 'しのぶ', '浩章', '友紀', '由喜', '空詩', '仁耀', '忠洋', '怜', '源喜', 
    '真里', '一毅', '勝哉', '国広', '芳也', '哲司', '翔吾', '輝明', '結', '千奈', '嘉弘', '勇介', '元明', '周二', '陽優', 
    '瑞穂', '孝司', '桃奈', 'こずえ', '昭男', '武志', 'ゆみ', '賢司', '和子', '卓也', '俊輔', '健次郎', '諒', '朝子', '有美', 
    '一馬', '慶子', '謙児', '裕美', '寛人', '晴美', '泉水', '素子', '祐奈', '久仁子', '毅', '宝姫', '隆洋', '祭', '久美子', 
    '勝則', '義一', '正典', '繁', '俊英', '紀克', '裕馬', '三幸', '祐太', '洋人', '英二', '将吾', '恵治', '佑香', '重行', 
    '紀美', '秀人', '太陽', '恒彦', '大志', '宏和', '悠', '芳徳', '佑実', '高志', '裕紀', '奨', 'なな', '勝生', '貴満', 
    '駿弥', '希一', '樹良々', '秀茉', '雄斗', '裕基', '啓治', '直美', '訓政', '実成', '弘次', '雅之', '幸哉', '健吾', '琳平', 
    '基裕', '貴', '政彦', '生', '紳弘', '正道', '光信', '禄仁', '真昭', '奈々', '光史', '賢二郎', '善己', '優弥', '隆成', 
    '佳史', '貴大', '輝', '猛', '大地', '輝男', '立樹', '伶音', '淳美', '英児', '美桜', '知弘', '橋蔵', '政行', '仁士', 
    '聡', '鉄也', '吉和', '藤太', '恵三', '舞', '直之', '理紗', '翔', '保生', '英数', '智紗衣', '貴士', '康夫', '隆', 
    '直哉', '浩治', '昂暁', '心吾', '純子', '洋平', '愛梨', '謙史朗', '良明', '莉奈', '憲行', '寛弥', '真人', '百合', '文俊', 
    '誓良', '真也', '重宣', '龍介', '武士', '清美', '達郎', '紳之亮', '秀之', '栄治', '知絵', '崚', '智史', '栄爾', '隼', 
    '健二', '隼人', '誠之', '毅一', '慎介', '雅裕', '友則', '晴人', '学志', '有理', '美緑', '悠花', '茂正', '健', '吉行', 
    '信雄', '大雅', '敏昭', '忠晴', '駿佑', '道', '均', '浩', '正司', '貴史', '聖人', '利夫', '与寛', '佳織', '三幹', 
    '哲之', '有裕', '政治', '孝博', '誠也', '桜', '悌二', '智之', '和人', '信樹', '寛法', '利徳', '央', 'まき', '芳美', 
    '梨沙', '昌義', '大道', '聡将', '可七実', '禎知', '光芳', '悦子', '博子', '信吾', '洋行', '雄紀', '祐輔', '浩之', '一洋', 
    '真由', '夏季', '邦夫', '司', '守嗣', '公則', '明宏', '達吉', '高史', '友理奈', '隆徳', '康介', '広孝', '恵里', '裕司', 
    '晃幸', '千明', '晴香', '敬悟', '秀三', '拓利', '透', '正侑', '頼宗', '兼輔', '基', '昭', '奈苗', '舞美', '正宗', 
    '孝征', '浩二', '敦揮', '響', '栄', '守成', '利幸', '浩人', '勝博', '俊二', '弓雄', '信二', '悟志', '明夫', '翔伍', 
    '芳弘', '孝太', '晃司', '将', '明生', '天馬', '金治', '利騰', '日富美', '日花里', 'ハリアント', '美和', '咲友理', '智稔', '潤二', 
    '照彦', '祐里子', '郁美', '才一郎', '武之', '真広', '淳', '功太', 'いづ美', '庸志', '涼', '雅俊', '寿里矢', '由美', '七海', 
    '強', '満', '信男', '正美', '修路', '一生', '文寧', '雅二', '慶', '秀和', '太亮', '芳恒', '晋', '重義', '二美子', 
    '勇作', '孝寿', '真央', '博貴', '知優', '萌', '栄二', '南', '知幸', 'めぐみ', '敦志', '翼', '公人', '洋一', 'みつよ', 
    '菜穂子', '英徳', '健司', '杏美', '正博', '光子', '祥昌', '彰二', '裕梨', '忠政', '潤', '泰輔', '昭生', '仁', '未唯', 
    '裕次', '亜由美', '耕治', '光弥', '準也', '翔磨', '智栄', '長松', '裕一', '修作', '和紀', '智仁', '雅友', '誉史明', '里美', 
    '理央', '奈津実', '清司', '貴浩', '昌子', '恵里那', '利明', '一樹', '隆義', '有夏', '隆之', '厚仁', '勝重', '伸幸', '健士郎', 
    '司朗', '三十志', '英之', '祥将', '幸茂', '隆行', '博崇', '尊', '啓三', '元親', '公靖', '尉宏', '淳士', '浩美', '繁洋', 
    '葵', '宗勝', '真希子', '亮蔵', '賀一', '恭輔', '良行', '君江', '元三', '里実', '久恵', '謙一', '伸介', '剛士', '利章', 
    '礼乃', '始', '依璃', '澄男', '真未', '普司', '東鉉', '智洋', '雅治', '俊彦', '正作', '真範', '泰久', '貴俊', '憲吾', 
    'ほのか', '智彰', '宏文', '将彦', '雄大', '伸之', '幹夫', '大', '光昭', '美智代', '耕作', '聖志', '節子', '兼信', '拓矢', 
    '厚子', '幸敏', '優生', '秀弥', '善明', '直樹', '香織', '史明', '朋実', '寛行', '裕茂', '猛志', '敦也', '元胤', '孝明', 
    '幸子', '紗奈', '雅史', '幸星', '開新', '清人', 'はやか', '雅雄', '翔太郎', '紀之', '啓二', '篤', '美帆', '健雄', '良春', 
    '咲希', '一矢', '公生', '佳典', '栄太', '四郎', '将太郎', '鉄平', '幸二', '沙里', '亜希子', '卓司', '雄哉', '駿平', '優一', 
    '康嗣', '晃史', '二郎', '恒一', '雅也', '幸之介', '祐作', '築夫', '治樹', '唯由', '雄', '龍貴', '二三男', '朋哉', '知典', 
    '潤子', '弥亘', '卓', '拓哉', '大智', '穂大', '卓郎', '芳顕', '祥之', '吉彦', '敏彦', '晋介', '彩香', '悠貴', '義盛', 
    '千穂', '裕絵', '悠介', '恒', '紗弥香', '雄一郎', '遼', '友宝', '雅道', '陽介', '拓美', '寛[典', '百恵', '重之', '繁輝', 
    '翔馬', '元気', '茂将', '真聖', '有希', '小鋒', '健之佑', '浩充', '智美', '道友', '和男', '晃広', '絵梨子', '千春', '智丈', 
    '雄二郎', '翔子', '奈月', '凌太朗', '星璃菜', '信明', '麻乃', '正倫', '友多加', '芳行', '由寛', '朋海', '時光', '雄基', '宏海', 
    '眞', '大樹', '芳久', '宅一', '龍紀', '恕生', '圭一', '鮎美', '肇也', '吉宏', '裕将', '由布子', '喜貴', '工幸', '秀二', 
    '喜春', '廣', '博訓', '敏雄', '康志郎', 'さやか', '成美', '一広', '大介', '真実', '雄人', '誉志郎', '正三', '昇平', '二千翔', 
    '晃朋', '小龍', '巌', '明憲', '翔之', 'テイエン', '直矢', '友良', '絵理香', '英典', '京介', '滉', '清春'
]

names1 = [x+"　"+y for x in list_last_name for y in list_first_name]
names2 = [x+" "+y for x in list_last_name for y in list_first_name]
names3 = [x+y for x in list_last_name for y in list_first_name]

def name_bg(chars=None, iters=25):
    return SymbolEmbedding.draw_text(
        height=800, width=800, iters=iters, color_init=[255, 255, 255], chars=chars, n_connect=1,
        range_scale=[15, 20], range_thickness=[1, 3], range_color=[0, 50],
        range_rotation=[-20, 20], is_PIL=True, font_pil=f"/{kkannotation.__path__[0]}/font/ipaexg.ttf",
        is_hanko=False, padding_loc=100, n_merge=1
    )

def name_label(chars=None, iters=25):
    imgs  = SymbolEmbedding.draw_text(
        height=800, width=800, iters=iters, color_init=[255, 255, 255], chars=chars, n_connect=1,
        range_scale=[15, 20], range_thickness=[1, 3], range_color=[0, 50],
        range_rotation=[-20, 20], is_PIL=True, font_pil=f"/{kkannotation.__path__[0]}/font/ipaexg.ttf",
        is_hanko=False, padding_loc=100, n_merge=1
    )
    imgs  = np.stack(imgs).astype(np.uint8)
    mask  = (imgs.min(axis=-1) < 250)
    label = np.array((["name" for _ in range(iters)]), dtype=object)
    return imgs, mask, label


if __name__ == "__main__":
    dirname = "./images_hanko"
    makedirs(dirname, exist_ok=True, remake=True)
    with open("./config_hanko.json") as f: config = json.load(f)
    emb = SymbolEmbedding(**config)
    emb.procs_draw.append(partial(name_bg, chars=names1, iters=10))
    emb.procs_draw.append(partial(name_bg, chars=names2, iters=10))
    emb.procs_draw.append(partial(name_bg, chars=names3, iters=10))
    emb.procs_draw.append(partial(name_bg, chars=list_last_name, iters=20))
    # emb.procs_label.append(partial(name_label, chars=names1, iters=10))
    # emb.procs_label.append(partial(name_label, chars=names2, iters=10))
    # emb.procs_label.append(partial(name_label, chars=names3, iters=10))
    # emb.procs_label.append(partial(name_label, chars=list_last_name, iters=20))
    Parallel(n_jobs=12, backend="loky", verbose=10)([
        delayed(lambda x, y: x.create_image(y, is_save=True))(
            emb, f"./{dirname}/train{i}.png"
        ) for i in range(args.num)
    ])
    coco = CocoManager()
    coco.add_jsons(glob.glob(f"./{dirname}/*.json"), root_dir=dirname)
    coco.organize_segmentation(min_point=6)
    coco.save("./coco_hanko.json", is_remove_tag=True)
