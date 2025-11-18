from backend.core.celery_app import celery_app
from backend.core.processing import STLAnalyzer, RhinoAnalyzer
import os
import logging

logger = logging.getLogger(__name__)

@celery_app.task(acks_late=True)
def process_analysis(reference_path: str, test_path: str, settings: dict):
    """
    Background task to process 3D analysis.
    """
    try:
        logger.info(f"Starting analysis for {test_path} against {reference_path}")
        
        # Determine analyzer type based on reference file extension
        if reference_path.lower().endswith('.3dm'):
            analyzer = RhinoAnalyzer()
            analyzer.load_reference_3dm(
                reference_path, 
                layer_weights=settings.get('layer_weights', {}),
                max_points=settings.get('max_points', 100000)
            )
            
            result = analyzer.process_test_file(
                file_path=test_path,
                stl_scale_to_mm=settings.get('stl_scale_to_mm', 1.0),
                use_global_reg=settings.get('use_global_reg', True),
                voxel_size=settings.get('voxel_size', 0.5),
                icp_threshold=settings.get('icp_threshold', 0.5),
                max_iter=settings.get('max_iter', 50),
                ignore_outside_bbox=settings.get('ignore_outside_bbox', True),
                include_notimportant_metrics=settings.get('include_notimportant_metrics', False),
                use_full_ref_global=settings.get('use_full_ref_global', False),
                icp_mode=settings.get('icp_mode', 'auto'),
                volume_voxel_size=settings.get('volume_voxel_size', 0.5)
            )
            
        else:
            analyzer = STLAnalyzer()
            analyzer.load_reference(
                reference_path,
                num_points=settings.get('num_points', 100000),
                layers=1,
                voxel_size=settings.get('voxel_size', 0.5)
            )
            
            # Add test file first (STLAnalyzer requires this step)
            analyzer.add_test_file(
                test_path,
                num_points=settings.get('num_points', 100000),
                nb_neighbors=settings.get('nb_neighbors', 20),
                std_ratio=settings.get('std_ratio', 2.0)
            )
            
            result = analyzer.process_test_file(
                file_path=test_path,
                use_global_reg=settings.get('use_global_reg', True),
                voxel_size=settings.get('voxel_size', 0.5),
                icp_threshold=settings.get('icp_threshold', 0.5),
                max_iter=settings.get('max_iter', 50),
                ignore_outside_bbox=settings.get('ignore_outside_bbox', True)
            )

        # Clean up result for JSON serialization (numpy arrays to lists)
        # Note: The refactored classes already handle some of this, but we ensure it here if needed.
        return {"status": "completed", "result": result}
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {"status": "failed", "error": str(e)}
