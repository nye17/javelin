from pyPdf import PdfFileWriter, PdfFileReader
import sys
import glob


ttaus = [20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
tnus  = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


def mergepdfs(inputf, outputf):
    output = PdfFileWriter()
    for document in inputf:
        print("adding...%s"%document)
        input1 = PdfFileReader(file(document, "rb"))
        # print the title of document1.pdf
        # print "title = %s" % (input1.getDocumentInfo().title)
        # add page 1 from input1 to output document, unchanged
        output.addPage(input1.getPage(0))
        # add page 2 from input1, but rotated clockwise 90 degrees
        # output.addPage(input1.getPage(1).rotateClockwise(90))
        # add page 3 from input1, rotated the other way:
        # output.addPage(input1.getPage(2).rotateCounterClockwise(90))
        # alt: output.addPage(input1.getPage(2).rotateClockwise(270))
        # add page 4 from input1, but first add a watermark from another pdf:
        # page4 = input1.getPage(3)
        # watermark = PdfFileReader(file("watermark.pdf", "rb"))
        # page4.mergePage(watermark.getPage(0))
        # add page 5 from input1, but crop it to half size:
        #page5 = input1.getPage(4)
        #page5.mediaBox.upperRight = (
        #    page5.mediaBox.getUpperRight_x() / 2,
        #    page5.mediaBox.getUpperRight_y() / 2
        #)
        #output.addPage(page5)
        # print how many pages input1 has:
        # print "document1.pdf has %s pages." % input1.getNumPages()
    # finally, write "output" to document-output.pdf
    print("saving...%s"%outputf)
    outputStream = file(outputf, "wb")
    output.write(outputStream)
    outputStream.close()


if __name__ == "__main__":    
    for ttau in ttaus:
        outputf = "ogle_example_T"+ str(ttau) + ".pdf"
        inputf = []
        for tnu in tnus:
            figname = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_hires.pdf"
            inputf.append(figname)
        mergepdfs(inputf, outputf)
    for tnu in tnus:
        outputf = "ogle_example_N"+ str(tnu) + ".pdf"
        inputf = []
        for ttau in ttaus:
            figname = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_hires.pdf"
            inputf.append(figname)
        mergepdfs(inputf, outputf)



