import CreateML
import Foundation

let trainingData = try MLDataTable(contentsOf: Bundle.main.url(forResource: "reviewsTrainingData", withExtension: "json")!)
let model = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")
try model.write(to: URL(fileURLWithPath: "/Users/zoufmehdi/Desktop/reviewsClassifier.mlmodel"))



let realDataFileURL = Bundle.main.url(forResource: "reviewsRealData", withExtension: "json")!
let realData = try MLDataTable(contentsOf: realDataFileURL)
model.evaluation(on: realData)




