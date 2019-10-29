#include <QApplication>
#include <QMainWindow>

#include "datasetdirwidget.h"

class LibrasCaptureMainWindow : public QMainWindow {
public:
    LibrasCaptureMainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        rscg::DatasetDirWidget *datasetDir = new rscg::DatasetDirWidget(this);
    }
};



int
main(int argc, char **argv) {

    QApplication app(argc, argv);

    LibrasCaptureMainWindow window;

    window.show();

    return app.exec();
}
