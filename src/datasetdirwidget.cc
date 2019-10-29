#include "datasetdirwidget.h"

#include <iostream>
#include <QHBoxLayout>
#include <QLabel>
#include <QFileDialog>

using namespace rscg;

namespace fs = boost::filesystem;

DatasetDirWidget::DatasetDirWidget(QWidget *parent,
                                   const std::string &datasetPath) 
    : 
    m_datasetPath(datasetPath),
    m_datasetNameLabel(new QLabel(datasetPath.c_str())),
    m_setDatasetDirBtn(new QPushButton("selecione diretorio da base de dados")),
    QWidget(parent) {
    QHBoxLayout *hbox = new QHBoxLayout(this);
    if (datasetPath == "") {
        m_datasetPath = "Dataset/";
        if (! checkDirExists(m_datasetPath)) {
            fs::path p(m_datasetPath);
            createAllNonExistentDirs(p / fs::path("sinais"));
        }
        m_datasetNameLabel->setText(m_datasetPath.c_str());
    }


    connect(m_setDatasetDirBtn, &QPushButton::clicked, this, 
            &DatasetDirWidget::changeDatasetDialogEv);

    hbox->addWidget(new QLabel("Diretorio da Base de Dados: "));
    hbox->addWidget(m_datasetNameLabel);
    hbox->addWidget(m_setDatasetDirBtn);
    
}

bool
DatasetDirWidget::checkDirExists(const fs::path &path) {
    return fs::is_directory(path);
}

bool
DatasetDirWidget::createDir(const fs::path &path) {
    std::cout << *path.begin() << std::endl;
    if (!checkDirExists(path)) {
        return fs::create_directory(path);
    }

    return true;
}

void
DatasetDirWidget::createAllNonExistentDirs(const fs::path &path) {
    std::cout << *path.begin() << std::endl;
    for (auto const &p : path) {
        std::cout << p << std::endl;
    }
}

void
DatasetDirWidget::changeDatasetDialogEv(int state) {
    auto dialogNameText = tr("Escolha o diretorio da base de dados");
    auto dialogOpts = QFileDialog::ShowDirsOnly | 
                      QFileDialog::DontResolveSymlinks;
    auto dir = QFileDialog::getExistingDirectory(this, dialogNameText, 
                                                 m_datasetPath.c_str(),
                                                 dialogOpts);
}