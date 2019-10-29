#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>

#include <string>
#include <boost/filesystem.hpp>

namespace rscg {

class DatasetDirWidget : public QWidget {
public:
    DatasetDirWidget(QWidget *parent = nullptr, 
                     const std::string &datasetPath = "");
private:
    //!
    //! checa se o diretorio existe
    //! \param path: localidade do diretorio
    //!
    bool checkDirExists(const boost::filesystem::path &path);
    
    //!
    //! cria o diretorio caso ele não exista
    //! \param path: localidade do diretorio
    //!
    bool createDir(const boost::filesystem::path &path);

    //!
    //! cria todos os diretorio inexistentes do caminho
    //! \param path: localidade do diretorio
    //!
    void createAllNonExistentDirs(const boost::filesystem::path &path);

    std::string m_datasetPath, m_newPathAux;
    QLabel *m_datasetNameLabel;
    QPushButton *m_setDatasetDirBtn;

    void changeDatasetDialogEv(int state);
};
}