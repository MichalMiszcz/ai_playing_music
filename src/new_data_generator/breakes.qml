import MuseScore 3.0

MuseScore {
      menuPath: "Plugins.BreakEvery4"
      description: "Dodaj łamanie systemu co 4 takty"
      version: "1.0"
      onRun: {
            var measuresPerSystem = 4;
            var cursor = curScore.newCursor();
            var measureCount = 0;

            cursor.rewind(0);

            while (cursor.segment) {
                  if (cursor.element && cursor.element.type === Element.MEASURE) {
                        if (measureCount > 0 && measureCount % measuresPerSystem === 0) {
                              var lb = newElement(Element.LAYOUT_BREAK);
                              lb.layoutBreakType = LayoutBreak.SYSTEM;
                              cursor.add(lb);
                        }
                        measureCount++;
                        cursor.nextMeasure();
                  } else {
                        cursor.next();
                  }
            }
            Qt.quit();
      }
}