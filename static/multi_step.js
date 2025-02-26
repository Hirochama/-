document.addEventListener('DOMContentLoaded', function(){
    const formSteps = document.querySelectorAll('.form-step');
    const progressSteps = document.querySelectorAll('.progress-step');
    const nextButtons = document.querySelectorAll('.next-btn');
    const prevButtons = document.querySelectorAll('.prev-btn');
    let currentStep = 0;

    // 各ステップの進捗表示を更新する関数
    function updateProgress() {
        progressSteps.forEach((step, index) => {
            if (index <= currentStep) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
    }

    // 「次へ」ボタンのクリック処理
    nextButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (currentStep < formSteps.length - 1) {
                formSteps[currentStep].classList.remove('active');
                currentStep++;
                formSteps[currentStep].classList.add('active');
                updateProgress();
            }
        });
    });

    // 「前へ」ボタンのクリック処理
    prevButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (currentStep > 0) {
                formSteps[currentStep].classList.remove('active');
                currentStep--;
                formSteps[currentStep].classList.add('active');
                updateProgress();
            }
        });
    });

    // 初期状態の進捗更新
    updateProgress();

    // エンターキーが押された際、フォーム送信を防いで「次へ」ボタンをクリックする
    const multiStepForm = document.getElementById('multiStepForm');
    if (multiStepForm) {
        multiStepForm.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault(); // エンターキーによる送信をキャンセル
                const currentStepForm = formSteps[currentStep];
                const nextButton = currentStepForm.querySelector('.next-btn');
                if (nextButton) {
                    nextButton.click();
                }
            }
        });
    }
});
