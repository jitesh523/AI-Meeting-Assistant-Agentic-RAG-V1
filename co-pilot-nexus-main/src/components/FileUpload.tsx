import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { 
  Upload, 
  File, 
  X, 
  FileText, 
  FileImage, 
  Loader2 
} from 'lucide-react';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface FileUploadProps {
  uploadedFiles: UploadedFile[];
  onFileUpload: (files: UploadedFile[]) => void;
  onFileRemove: (id: string) => void;
}

export const FileUpload = ({ uploadedFiles, onFileUpload, onFileRemove }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return FileImage;
    return FileText;
  };

  const handleFileUpload = useCallback(async (files: FileList) => {
    setIsUploading(true);
    
    const newFiles: UploadedFile[] = [];
    
    for (const file of Array.from(files)) {
      // Validate file type
      const allowedTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'image/jpeg',
        'image/png'
      ];
      
      if (!allowedTypes.includes(file.type)) {
        toast.error(`File type ${file.type} not supported`);
        continue;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error(`File ${file.name} is too large (max 10MB)`);
        continue;
      }

      // Simulate upload delay
      await new Promise(resolve => setTimeout(resolve, 500));

      const uploadedFile: UploadedFile = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        name: file.name,
        size: file.size,
        type: file.type
      };

      newFiles.push(uploadedFile);
    }

    if (newFiles.length > 0) {
      onFileUpload([...uploadedFiles, ...newFiles]);
      toast.success(`${newFiles.length} file(s) uploaded successfully`);
    }

    setIsUploading(false);
  }, [uploadedFiles, onFileUpload]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files);
    }
  }, [handleFileUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files);
    }
  };

  const handleRemoveFile = (id: string) => {
    onFileRemove(id);
    toast.success('File removed');
  };

  return (
    <div className="space-y-4">
      {/* Upload Zone */}
      <div
        className={`upload-zone ${isDragging ? 'border-primary bg-primary/5' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {isUploading ? (
          <div className="flex items-center justify-center gap-2">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <span className="text-muted-foreground">Uploading...</span>
          </div>
        ) : (
          <>
            <Upload className="h-8 w-8 text-muted-foreground mb-2 mx-auto" />
            <p className="text-muted-foreground mb-2">
              Drag & drop files here, or{' '}
              <label className="text-primary cursor-pointer hover:text-primary-glow">
                browse
                <input
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.jpg,.jpeg,.png"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>
            </p>
            <p className="text-xs text-muted-foreground">
              Supports PDF, DOCX, TXT, JPG, PNG • Max 10MB each
            </p>
          </>
        )}
      </div>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-card-foreground">
            Uploaded Files ({uploadedFiles.length})
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {uploadedFiles.map((file) => {
              const FileIcon = getFileIcon(file.type);
              
              return (
                <div
                  key={file.id}
                  className="flex items-center gap-3 p-2 rounded-lg bg-muted/20 border border-border"
                >
                  <FileIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-card-foreground truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleRemoveFile(file.id)}
                    className="h-6 w-6 p-0 hover:bg-danger/10 hover:text-danger flex-shrink-0"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};