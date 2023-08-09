//
//  Predictor.swift
//  boxingText
//
//  Created by Seljuq Haider  on 6/22/23.
//

import Foundation
import Vision

typealias ThrowingClassifier =  boxingPunchTestClassifier_1

protocol PredictorDelegate: AnyObject {
    func predictor(_ predictor: Predictor, didFindNewRecognizedPoints points: [CGPoint])
    func predictor(_ predictor: Predictor, didLabelAction action : String, with confidence: Double)
}

class Predictor {
    
    weak var delegate: PredictorDelegate?
    
    let predictionWindowSize = 30 //changes the amount of frames for prediction
    var posesWindow: [VNHumanBodyPoseObservation] = []
    
    init() {
        posesWindow.reserveCapacity(predictionWindowSize)
    }
    
    func estimation(sampleBuffer : CMSampleBuffer) {
        let requestHandler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer,
                                                   orientation: .up)
        
        let request = VNDetectHumanBodyPoseRequest(completionHandler: bodyPoseHandler)
        
        do {
            try requestHandler.perform([request])
        } catch {
            print("Unable to perform the request, with error:  \(error)")
        }
    }
    
    func bodyPoseHandler(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNHumanBodyPoseObservation] else { return }
        
        observations.forEach {
            processObservation($0)
        }
        
        if let result = observations.first {
            storeObservation(result)
            
            labelActionType()
        }
    }
    
    func labelActionType() {
        guard let throwingClassifier = try? ThrowingClassifier(configuration: MLModelConfiguration()),
              let poseMultiArray = prepareInputWithObservations(posesWindow),
              let predictions = try? throwingClassifier.prediction(poses: poseMultiArray) else {
            return
        }
        
        let label = predictions.label
        let confidence = predictions.labelProbabilities[label] ?? 0
        
        delegate?.predictor(self, didLabelAction: label, with: confidence)
    }
    
    func prepareInputWithObservations(_ observations: [VNHumanBodyPoseObservation]) -> MLMultiArray? {
        let numAvailiableFrames = observations.count
        let observationsNeeded = 30
        var multiArrayBuffer = [MLMultiArray]()
        
        for frameIndex in 0 ..< min(numAvailiableFrames, observationsNeeded) {
            let pose = observations[frameIndex]
            do {
                let oneFrameMultiArray = try pose.keypointsMultiArray()
                multiArrayBuffer.append(oneFrameMultiArray)
            } catch {
                continue
            }
        }
        
        if numAvailiableFrames < observationsNeeded {
            for _ in 0 ..< (observationsNeeded - numAvailiableFrames) {
                do {
                    let oneFrameMultiArray = try MLMultiArray(shape: [1, 3, 18], dataType: .double)
                    try resetMultiArray(oneFrameMultiArray)
                    multiArrayBuffer.append(oneFrameMultiArray)
                } catch {
                    continue
                }
            }
        }
        
        return MLMultiArray(concatenating: [MLMultiArray](multiArrayBuffer), axis: 0, dataType: .float)
    }
    
    func resetMultiArray(_ predictionWindow: MLMultiArray, with value: Double = 0.0) throws {
        let pointer = try UnsafeMutableBufferPointer<Double>(predictionWindow)
        pointer.initialize(repeating: value)
    }
    
    func storeObservation(_ observation: VNHumanBodyPoseObservation) {
        if posesWindow.count >= predictionWindowSize {
            posesWindow.removeFirst()
        }
        
        posesWindow.append(observation)
    }
    
    func processObservation(_ observation: VNHumanBodyPoseObservation) {
        do {
            let recgonizedPoints = try observation.recognizedPoints(forGroupKey: .all)
            
            let displayPoints = recgonizedPoints.map {
                CGPoint(x: $0.value.x , y: 1 - $0.value.y)
            }
            
            delegate?.predictor(self, didFindNewRecognizedPoints: displayPoints)
        } catch {
            print("Error finding recgonized points")
        }
    }
    
}

