import UIKit

final class ViewController: UIViewController {
    private let resultLabel = UILabel()
    private let fpsLabel = UILabel()
    private var pipeline: MobileNetV2Pipeline?

    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .black

        let pipeline = MobileNetV2Pipeline(frame: view.bounds)
        self.pipeline = pipeline

        let previewView = pipeline.previewView
        previewView.frame = view.bounds
        previewView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(previewView)

        resultLabel.frame = CGRect(x: 0, y: 40, width: view.bounds.width, height: 60)
        resultLabel.textColor = .white
        resultLabel.font = .boldSystemFont(ofSize: 15)
        resultLabel.textAlignment = .center
        resultLabel.numberOfLines = 2
        resultLabel.autoresizingMask = [.flexibleWidth, .flexibleBottomMargin]
        view.addSubview(resultLabel)

        fpsLabel.frame = CGRect(x: 0, y: view.bounds.height - 100, width: view.bounds.width, height: 50)
        fpsLabel.textColor = .white
        fpsLabel.font = .boldSystemFont(ofSize: 13)
        fpsLabel.textAlignment = .center
        fpsLabel.autoresizingMask = [.flexibleWidth, .flexibleTopMargin]
        view.addSubview(fpsLabel)

        pipeline.predictionHandler = { [weak self] label, confidence, fps in
            guard let self else { return }
            self.fpsLabel.text = "\(fps) FPS"

            guard let label else {
                self.resultLabel.text = nil
                return
            }

            let percentage = Int((confidence * 100).rounded())
            self.resultLabel.text = "\(label) (\(percentage)%)"
        }

        view.bringSubviewToFront(resultLabel)
        view.bringSubviewToFront(fpsLabel)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        pipeline?.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        pipeline?.stop()
    }

    override var prefersStatusBarHidden: Bool {
        true
    }
}
