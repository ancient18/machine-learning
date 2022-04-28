/* 
	@decription: 决策树

	@param: {
		config:包	含数据集的一些信息;
		items:数据集每一行的信息;
		attr:标签属性;
		counter:统计属性值在数据集中的次数的对象
	}

*/

class DecisionTree {
  constructor(config) {
    this._predicates = {
      //分割函数
      "==": function (a, b) {
        return a == b;
      }, //针对非数字值的比较
      ">=": function (a, b) {
        return a >= b;
      }, //针对数值的比较
    };
    if (Object.prototype.toString.call(config) === "[object Object]")
      this.training(config);
  }

  //统计属性值在数据集中的次数
  countUniqueValues(items, attr) {
    const counter = {};
    for (let i of items) {
      if (!counter[i[attr]]) counter[i[attr]] = 0;
      counter[i[attr]]++;
    }
    return counter;
  }

  //寻找最频繁的特定属性值
  mostFrequentValue(items, attr) {
    const counter = this.countUniqueValues(items, attr);

    //获取对象中值最大的Key  假设 counter={a:9,b:2} 得到 "a"
    let mostFrequentValue;
    for (let i in counter) {
      if (!mostFrequentValue) mostFrequentValue = i;
      if (counter[i] > counter[mostFrequentValue]) mostFrequentValue = k;
    }
    return mostFrequentValue;
  }

  //根据属性切割数据集
  split(items, attr, predicate, pivot) {
    const data = {
      match: [], //适合的数据集
      notMatch: [], //不适合的数据集
    };
    for (let item of items) {
      //遍历训练集
      if (predicate(item[attr], pivot)) {
        //比较是否满足条件
        data.match.push(item);
      } else {
        data.notMatch.push(item);
      }
    }
    return data;
  }

  //计算熵
  entropy(items, attr) {
    const counter = this.countUniqueValues(items, attr); //计算值的出现数
    let p,
      entropy = 0; //H(S)=entropy=∑(P(Xi)(log2(P(Xi))))
    for (let i in counter) {
      //entropy+=-(P(Xi)(log2(P(Xi))))
      p = counter[i] / items.length; //P(Xi)概率值
      entropy += -p * Math.log2(p);
    }
    return entropy;
  }

  buildDecisionTree(config) {
    const trainingSet = config.trainingSet; //训练集
    const minItemsCount = config.minItemsCount; //训练集项数
    const categoryAttr = config.categoryAttr; //用于区分的类别属性
    const entropyThrehold = config.entropyThrehold; //熵阈值
    const maxTreeDepth = config.maxTreeDepth; //递归深度
    const ignoredAttributes = config.ignoredAttributes; //忽略的属性
    // 树最大深度为0 或训练集的大小 小于指定项数 终止树的构建过程
    if (maxTreeDepth == 0 || trainingSet.length <= minItemsCount) {
      return { category: this.mostFrequentValue(trainingSet, categoryAttr) };
    }
    //初始计算 训练集的熵
    let initialEntropy = this.entropy(trainingSet, categoryAttr); //<===H(S)
    //训练集熵太小 终止
    if (initialEntropy <= entropyThrehold) {
      return { category: this.mostFrequentValue(trainingSet, categoryAttr) };
    }
    let alreadyChecked = []; //标识已经计算过了

    let bestSplit = { gain: 0 }; //储存当前最佳的分割节点数据信息
    //遍历数据集
    for (let item of trainingSet) {
      // 遍历项中的所有属性
      for (let attr in item) {
        //跳过区分属性与忽略属性
        if (attr == categoryAttr || ignoredAttributes.indexOf(attr) >= 0)
          continue;
        let pivot = item[attr]; // 当前属性的值
        let predicateName = typeof pivot == "number" ? ">=" : "=="; //根据数据类型选择判断条件
        let attrPredPivot = attr + predicateName + pivot;
        if (alreadyChecked.indexOf(attrPredPivot) >= 0) continue; //已经计算过则跳过
        alreadyChecked.push(attrPredPivot); //记录
        let predicate = this._predicates[predicateName]; //匹配分割方式
        let currSplit = this.split(trainingSet, attr, predicate, pivot);
        let matchEntropy = this.entropy(currSplit.match, categoryAttr); //  H(match) 计算分割后合适的数据集的熵
        let notMatchEntropy = this.entropy(currSplit.notMatch, categoryAttr); // H(on match) 计算分割后不合适的数据集的熵
        //计算信息增益:
        // IG(A,S)=H(S)-(∑P(t)H(t)))
        // t为分裂的子集match(匹配),on match(不匹配)
        // P(match)=match的长度/数据集的长度
        // P(on match)=on match的长度/数据集的长度
        let iGain =
          initialEntropy -
          (matchEntropy * currSplit.match.length +
            notMatchEntropy * currSplit.notMatch.length) /
            trainingSet.length;
        //不断匹配最佳增益值对应的节点信息
        if (iGain > bestSplit.gain) {
          bestSplit = currSplit;
          bestSplit.predicateName = predicateName;
          bestSplit.predicate = predicate;
          bestSplit.attribute = attr;
          bestSplit.pivot = pivot;
          bestSplit.gain = iGain;
        }
      }
    }

    // 找不到最优分割
    if (!bestSplit.gain) {
      return { category: this.mostFrequentValue(trainingSet, categoryAttr) };
    }
    // 递归绑定子树枝
    config.maxTreeDepth = maxTreeDepth - 1; //减小1深度
    config.trainingSet = bestSplit.match; //将切割 match 训练集作为下一节点的训练集
    let matchSubTree = this.buildDecisionTree(config); //递归匹配子树节点
    config.trainingSet = bestSplit.notMatch; //将切割 notMatch 训练集作为下一节点的训练集
    let notMatchSubTree = this.buildDecisionTree(config); //递归匹配子树节点
    return {
      attribute: bestSplit.attribute,
      predicate: bestSplit.predicate,
      predicateName: bestSplit.predicateName,
      pivot: bestSplit.pivot,
      match: matchSubTree,
      notMatch: notMatchSubTree,
      matchedCount: bestSplit.match.length,
      notMatchedCount: bestSplit.notMatch.length,
    };
  }

  training(config) {
    this.root = this.buildDecisionTree({
      trainingSet: config.trainingSet, //训练集
      ignoredAttributes: config.ignoredAttributes || [], // 被忽略的属性比如:姓名、名称之类的
      categoryAttr: config.categoryAttr || "category", //用于区分的类别属性
      minItemsCount: config.minItemsCount || 1, //最小项数量
      entropyThrehold: config.entropyThrehold || 0.01, //熵阈值
      maxTreeDepth: config.maxTreeDepth || 70, //递归的最大深度
    });
  }

  //预测 测试
  predict(data) {
    let attr, value, predicate, pivot;
    let tree = this.root;
    while (true) {
      if (tree.category) {
        return tree.category;
      }
      attr = tree.attribute;
      value = data[attr];
      predicate = tree.predicate;
      pivot = tree.pivot;
      if (predicate(value, pivot)) {
        tree = tree.match;
      } else {
        tree = tree.notMatch;
      }
    }
  }
}

const fs = require("fs");
const readline = require("readline");
const path = require("path");

//调用方法
const fpath = path.resolve("./", "decisonTree/Iris/iris.data");
read_file(fpath, function (data) {
  console.log(data);
});

let sepalLength, sepalWieth, petalLength, petalWidth, className;

// 定义读取方法;
function read_file(path, callback) {
  const fRead = fs.createReadStream(path);
  const objReadline = readline.createInterface({
    input: fRead,
  });

  const arr = new Array();
  objReadline.on("line", function (line) {
    const lineArr = line.split(",");
    [sepalLength, sepalWieth, petalLength, petalWidth, className] = lineArr;
    arr.push({ sepalLength, sepalWieth, petalLength, petalWidth, className });
  });

  objReadline.on("close", function () {
    const decisionTree = new DecisionTree({
      trainingSet: arr, //训练集
      categoryAttr: "className", //用于区分的类别属性
      // ignoredAttributes: ["姓名"], //被忽略的属性
    });

    const comic = {
      sepalLength: 6.7,
      sepalWieth: 3.0,
      petalLength: 5.2,
      petalWidth: 2.3,
    };

    console.log(comic, decisionTree.predict(comic));
    console.log(arr);
  });
}
